
if __name__ == "__main__":
    pass
"""
    with open(OUTPUT_FILENAME, "rb") as rf:
        soln = pickle.load(rf)

    variables = soln["x"]

    # Extract variables from the solution:
    o = 0
    
    qv = [np.array(variables[o + idx * (robot.nq - 1) : o + (idx + 1) * (robot.nq - 1)]) for idx in range(N_KNOTS)]
    o += N_KNOTS * (robot.nq - 1)

    vv = [np.array(variables[o + idx * robot.nv : o + (idx + 1) * robot.nv]) for idx in range(N_KNOTS)]
    o += N_KNOTS * robot.nv
    
    av = [np.array(variables[o + idx * robot.nv : o + (idx + 1) * robot.nv]) for idx in range(N_KNOTS)]
    o += N_KNOTS * robot.nv

    # tau
    o += N_KNOTS * len(actuated_joints)

    λs = unflatten(variables[o : o + 12 * N_KNOTS], (4, 3))
    o += N_KNOTS * len(FEET) * 3

    # q_s, v_s, a_s = ca.SX.sym("q", 18, 1), ca.SX.sym("v", 18, 1), ca.SX.sym("a", 18, 1)
    # fkv = ca.Function("fkv", [q_s, v_s, a_s], [fk(q_s, v_s, a_s)[1]])
    # fkx = ca.Function("fkv", [q_s, v_s, a_s], [fk(q_s, v_s, a_s)[0]])

    # pos_hist, vel_hist = [], []
    # err = []

    # for k in range(N_KNOTS):
    #     pos = fkx(qv[k], vv[k], av[k])[0, :]
    #     vel = fkv(qv[k], vv[k], av[k])[0, :]

    #     if (k > 0):
    #         integr = pos_hist[-1] + DELTA_T * vel_hist[-1]
    #         err.append(pos - integr)

    #     pos_hist.append(pos)
    #     vel_hist.append(vel)
        
    #     # print("pos:", pos, "\t\tvel:", vel)
    
    # print(np.sum(np.abs(err), axis = 0))

    # # With 30 Hz:
    # # LOCAL_WORLD_ALIGNED: [[0.00219949 0.00270747 0.00322794]]
    # # LOCAL: [[0.04122901 0.00220468 0.08294682]]
    # # WORLD: [[0.21330099 0.09794994 0.27863034]]
    # exit()

    # print(qv)
    # print(λs)

    input("Press ENTER to play trajectory...")
    
        TODO: While the constraint violation is as follows below,
        integrating manually the positions results in bad movement.

        The robot flies, and the legs move off the ground.
        ALSO, there's a difference between pinocchio integrate and custom

        TODO: Once you figure this out, you have to make sure that FK_vel_z = 0
        if it _actually_ looks like it is...
        If you remove that constraint everything optimizes so, LOCAL_WORLD_ALIGNED
        might not be correct. Have a look.


    # q_mrp = qv[0]
    # q = ca_to_np(q_mrp_to_quat(qv[0].T[0]))

    for idx, q_mrp in tqdm(enumerate(qv[:-1])):
        # vel_avg = 0.5 * (vv[idx] + vv[idx+1])
        # q = pin.integrate(robot.model, q,  vv[idx] * DELTA_T)
        # q = ca_to_np(q_mrp_to_quat(integrate_custom(q_mrp, vel_avg * DELTA_T)))
        q = ca_to_np(q_mrp_to_quat(qv[idx]))
        robot.display(q)
        time.sleep(DELTA_T)
        # input()
    """