from collections import defaultdict
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from constraints import *
from utilities import flatten_mats
from continuity import ContinuityInfo
from variables import CollocationVars
from transcription import Subproblem, TranscriptionInfo

@dataclass(frozen = True)
class Solution:
    # Solver output containing the objective value, decision variable
    # assignments and Lagrange multipliers:
    solver_output:          dict

    # Transcription information for all subproblems:
    transcription_infos:    list[TranscriptionInfo]

@dataclass
class Problem:
    subproblems:            list[Subproblem]
    continuity_info:        list[ContinuityInfo]  = field(default_factory = list)

    transcribed:            bool                  = field(default = False, init = False)
    problem_constraints:    list[Constraint]      = field(default_factory = list, init = False)

    def __post_init__(self):
        assert len(self.continuity_info) == len(self.subproblems) - 1, \
            "Continuity info instances must be exactly one fewer than the number of subproblems."
        
    def transcribe(self):
        if self.transcribed:
            raise RuntimeError("Problem has already been transcribed!")

        transcribed_subps = set()

        for idx, subp in enumerate(self.subproblems):
            if subp.name in transcribed_subps:
                raise RuntimeError("Problem can't contain a duplicate subproblem name!")
            
            print(f"Transcribing subproblem '{subp.name}' using {subp.n_knots} knots...")

            subp.transcribe(is_subsequent = (idx > 0))
            transcribed_subps.add(subp.name)

        # Add continuity constraints between subproblems:
        for p_idx in range(len(self.subproblems) - 1):
            cont_info   = self.continuity_info[p_idx]
            vars_before = self.subproblems[p_idx].dvars.get_vars_at_knot(-1)
            vars_after  = self.subproblems[p_idx + 1].dvars.get_vars_at_knot(0)

            self.problem_constraints.extend(
                cont_info.create_constraints(vars_before, vars_after)
            )

        self.transcribed = True

    @property
    # Collects variables from all subproblems, as well as their bounds.
    # Returns column vectors corresponding to: x, lb, ub.
    def variables(self) -> tuple[ca.SX, np.ndarray, np.ndarray]:
        assert self.transcribed

        flat_vars      = []    
        bounds_by_name = defaultdict(lambda: (-ca.inf, ca.inf))
        
        for subp in self.subproblems:
            flat_vars.append(subp.dvars.flatten())

            # Intersect all bounds that correspond to the same variable:
            for b in filter(lambda ct: isinstance(ct, Bound), subp.constraints):
                for b_var in ca_iter(b.variables):
                    prev_b = bounds_by_name[b_var.name()]

                    bounds_by_name[b_var.name()] = (
                        max(prev_b[0], b.lb),
                        min(prev_b[1], b.ub)
                    )

        # Combine variables and bounds across all subproblems:        
        vars = ca.vertcat(*flat_vars)
        assert vars.shape[1] == 1

        lbs, ubs = [], []
        for v in ca_iter(vars):
            lb, ub = bounds_by_name[v.name()]
            lbs.append(lb); ubs.append(ub)

        return vars, np.vstack(lbs), np.vstack(ubs)

    @property
    # Collects constrained expressions for all subproblems, including
    # problem-level continuity ones. Returns column vectors of
    # the expressions, lower and upper bounds.
    def constraints(self) -> tuple[ca.SX, np.ndarray, np.ndarray]:
        assert self.transcribed
        exprs, lbs, ubs = [], [], []

        for subp in self.subproblems:
            for c in filter(lambda ct: isinstance(ct, Constraint), subp.constraints):
                exprs.append(c.expr); lbs.append(c.lb), ubs.append(c.ub)
        
        for prob_c in self.problem_constraints:
            exprs.append(prob_c.expr); lbs.append(prob_c.lb), ubs.append(prob_c.ub)

        return flatten_mats(exprs), flatten_mats(lbs), flatten_mats(ubs)

    @property
    # Collects complementarity constraints for all problems. Returns a list of
    # indices in the variable vector as expected by the Knitro interface.
    def complementarities(self) -> list[tuple[int, int]]:
        assert self.transcribed
        var_idxs_by_name = {}

        # Store variable indices in the global problem vector indexed by
        # the variable name so we can find the variables quickly:
        for v_idx, v in enumerate(ca_iter(self.variables[0])):
            var_idxs_by_name[v.name()] = v_idx

        compl_idxs = []

        for subp in self.subproblems:
            for compl in filter(lambda ct: isinstance(ct, Complementarity), subp.constraints):
                compl_idxs.append((
                    var_idxs_by_name[compl.var1],
                    var_idxs_by_name[compl.var2],
                ))
        
        return compl_idxs
    
    # Returns the problem-wide objective. That is a weighted average of subproblem
    # objectives based on their time duration.
    @property
    def objective(self) -> ca.SX:
        # TODO: This is slightly inaccurate since the first knot of
        #       all subproblems after the first one always repeats the
        #       last knot of its previous subproblem (ie. there's one
        #       fewer Δt for each subproblem after the first).
        total_duration = sum(s.task.duration for s in self.subproblems)

        return sum(
            float(subp.task.duration / total_duration) * subp.objective
            for subp in self.subproblems
        )
    
    def guess(self) -> np.ndarray:
        g_vars = [sp.create_guess().flatten() for sp in self.subproblems]
        return np.vstack(g_vars)

    @staticmethod
    # Utility method to combine trajectories from multiple subproblems into
    # a single trajectory. This is useful when loading a solution to a problem
    # composed of multiple subproblems.
    # The last knot of every subproblem (excl. last subproblem) should represent
    # the same time as the first knot of its subsequent subproblem. Therefore,
    # the stitched trajectory will drop these initial knots for all subproblems
    # after the first.
    def stitch_trajectories(trajectories: list[CollocationVars[np.ndarray]]) \
        -> CollocationVars[np.ndarray]:

        # Total knot count should be the sum of all subproblem knot counts,
        # excluding duplicated points:
        n_subp = len(trajectories)
        total_knot_count = sum(t.n_knots for t in trajectories) - (n_subp - 1)
        global_solution = CollocationVars[np.ndarray](total_knot_count)

        for idx, traj in enumerate(trajectories):
            # Skip first knot for all subproblems except the first:
            for k in range(0 if idx == 0 else 1, traj.n_knots):
                global_solution.append_knot(
                    kvars = traj.get_vars_at_knot(k),
                    duration = traj.knot_duration[k]
                )

            # Append all slack variables from the subproblem, in case
            # they're needed:
            global_solution.slack_vars.extend(traj.slack_vars)        

        return global_solution
    
    @staticmethod
    # Utility method to load subproblem trajectories from a problem solution:
    def load_solution(soln: Solution) -> list[CollocationVars[np.ndarray]]:
        vec = soln.solver_output["x"]
        subp_solns, cur_vec_offset = [], 0

        for info in soln.transcription_infos:
            # Load variables for each subproblem, starting at the
            # current vector offset:
            subp_soln, var_count = CollocationVars[np.ndarray].unflatten(
                info.n_knots, info.slack_var_count, info.dt, vec[cur_vec_offset:]
            )

            subp_solns.append(subp_soln)

            # Increase offset in the global vector for the next subproblem:
            cur_vec_offset += var_count

        return subp_solns