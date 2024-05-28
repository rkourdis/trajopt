        #########################

        N_KNOTS_LOW_FREQ = int(TRAJ_DURATION * 100)
        with open("solution_100hz_2sec.bin", "rb") as rf:
            soln = pickle.load(rf)["x"]

        # Extract variables from the solution:
        o = 0
        
        q_lowfreq = unflatten(soln[o : o + 18 * N_KNOTS_LOW_FREQ], (18, 1))
        o += N_KNOTS_LOW_FREQ * 18
        
        v_lowfreq = unflatten(soln[o : o + 18 * N_KNOTS_LOW_FREQ], (18, 1))
        o += N_KNOTS_LOW_FREQ * 18

        a_lowfreq = unflatten(soln[o : o + 18 * N_KNOTS_LOW_FREQ], (18, 1))
        o += N_KNOTS_LOW_FREQ * 18

        τ_lowfreq = unflatten(soln[o : o + 12 * N_KNOTS_LOW_FREQ], (12, 1))
        o += N_KNOTS_LOW_FREQ * 12
        
        λ_lowfreq = unflatten(soln[o : o + 12 * N_KNOTS_LOW_FREQ], (4, 3))
        o += N_KNOTS_LOW_FREQ * len(FEET) * 3

        f_pos_lowfreq = unflatten(soln[o : o + 12 * N_KNOTS_LOW_FREQ], (4, 3))
        o += N_KNOTS_LOW_FREQ * len(FEET) * 3

        q_g  = chain.from_iterable([[q_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        v_g  = chain.from_iterable([[v_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        a_g  = chain.from_iterable([[a_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        τ_g  = chain.from_iterable([[τ_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        λ_g  = chain.from_iterable([[λ_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        f_pos_g  = chain.from_iterable([[f_pos_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])

        # q_g  = [initial_state.q for _ in range(N_KNOTS)]
        # v_g  = [initial_state.v for _ in range(N_KNOTS)]
        # a_g  = [np.zeros((robot.nv, 1)) for _ in range(N_KNOTS)]
        # τ_g  = [np.copy(tau0) for _ in range(N_KNOTS)]
        # λ_g  = [np.copy(λ0) for _ in range(N_KNOTS)]
        # f_pos_g = [np.array(num_fk(q_g[0], v_g[0], a_g[0])[0]) for _ in range(N_KNOTS)]
        ########################
