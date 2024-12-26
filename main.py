import pulp

def solve_enclosing_zonotope_2d(points, vectors):
    """
    2次元版の Discrete Oriented Enclosing Zonotope を
    線形計画法で解くサンプル実装。

    引数:
        points : list of tuple ( (x_j, y_j), ... ) の点集合
        vectors: list of tuple ( (vx_i, vy_i), ... ) の単位ベクトル集合

    戻り値:
        (p, c_list, b_dict) のタプル
        p       = (px, py)
        c_list  = [c1, c2, ..., ck]
        b_dict  = {(i,j): (b_{i,j}), ...}
    """
    n = len(points)
    k = len(vectors)

    # ==============
    # 変数の定義
    # ==============

    # p = (px, py) ・・・ ゾノトープの平行移動ベクトル
    px = pulp.LpVariable('px', cat=pulp.LpContinuous)
    py = pulp.LpVariable('py', cat=pulp.LpContinuous)

    # c_i >= 0 ・・・ 各ジェネレーター方向の幅を決めるスカラー
    c = [pulp.LpVariable(f'c_{i}', lowBound=0, cat=pulp.LpContinuous) for i in range(k)]

    # b_{i,j} ・・・ 中間変数, -c_i <= b_{i,j} <= c_i
    # （ただし PuLP は直接 "変数の上下限が別の変数" の形式は書けないため，
    #   線形制約に分解して設定する）
    b = {}
    for i in range(k):
        for j in range(n):
            b_ij = pulp.LpVariable(f'b_{i}_{j}', lowBound=None, upBound=None, cat=pulp.LpContinuous)
            b[(i, j)] = b_ij

    # ==============
    # 目的関数
    # ==============
    # min \sum_{i=1}^k c_i
    problem = pulp.LpProblem("Discrete_Oriented_Enclosing_Zonotope_2D", pulp.LpMinimize)
    problem += pulp.lpSum([c[i] for i in range(k)]), "Minimize_sum_of_c"

    # ==============
    # 制約の設定
    # ==============

    # 1) 各点 p_j = p + sum_{i=1}^k (b_{i,j} * v_i)
    #    x座標, y座標それぞれに対して等式制約
    for j in range(n):
        x_j, y_j = points[j]

        # x座標:  x_j = px + Σ b_{i,j} * vx_i
        problem += (
            x_j
            == px + pulp.lpSum([b[(i, j)] * vectors[i][0] for i in range(k)])
        ), f"constraint_x_{j}"

        # y座標:  y_j = py + Σ b_{i,j} * vy_i
        problem += (
            y_j
            == py + pulp.lpSum([b[(i, j)] * vectors[i][1] for i in range(k)])
        ), f"constraint_y_{j}"

    # 2) -c_i <= b_{i,j} <= c_i
    #    これは下記2つの不等式に分ける:
    #       b_{i,j} + c_i >= 0  ( => b_{i,j} >= -c_i )
    #       c_i - b_{i,j} >= 0  ( => b_{i,j} <= c_i )
    for i in range(k):
        for j in range(n):
            problem += b[(i, j)] + c[i] >= 0,  f"lower_bound_{i}_{j}"
            problem += c[i] - b[(i, j)] >= 0,  f"upper_bound_{i}_{j}"

    # ==============
    # ソルバー実行
    # ==============
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # 解の取得
    px_val = pulp.value(px)
    py_val = pulp.value(py)
    c_vals = [pulp.value(ci) for ci in c]
    b_vals = {}
    for key, var in b.items():
        b_vals[key] = pulp.value(var)

    return (px_val, py_val), c_vals, b_vals


if __name__ == "__main__":
    # --------------------------------------------------------
    # 例: 点3つと，単位ベクトル2つを与える
    #     これらを「離散的ゾノトープ」で覆うような c_i, p を求める
    # --------------------------------------------------------
    points = [
        (0.0, 2.0),
        (3.0, 2.0),
        (1.0, 0.0)
    ]
    # 単位ベクトル2つ (例: x方向，y方向)
    vectors = [
        (1.0, 0.0),
        (0.0, 1.0)
    ]

    p_sol, c_sol, b_sol = solve_enclosing_zonotope_2d(points, vectors)

    print("========== 結果 ==========")
    print(f"p = ({p_sol[0]:.3f}, {p_sol[1]:.3f})")
    for i, val in enumerate(c_sol):
        print(f"c_{i+1} = {val:.3f}")

    # b_{i,j} は制限付きの中間変数（-c_i <= b_{i,j} <= c_i）
    # 必要であれば確認
    # for (i, j), val in b_sol.items():
    #     print(f"b_({i},{j}) = {val:.3f}")
