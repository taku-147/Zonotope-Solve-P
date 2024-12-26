/*
 * Discrete Oriented Enclosing Zonotope (2D版) を
 * GLPK を用いて線形計画問題として解くサンプル。
 *
 * 変数:
 *   p_x, p_y        (ゾノトープ平行移動ベクトルの x, y)
 *   c_i (i=1..k)    (各方向ベクトルのスカラー幅, 非負)
 *   b_{i,j}         (中間変数, -c_i <= b_{i,j} <= c_i)
 *
 * 制約:
 *   1) 各点 p_j = p + Σ b_{i,j} * v_i
 *      → x座標, y座標 それぞれ等式で表現
 *
 *   2) -c_i <= b_{i,j} <= c_i
 *      → b_{i,j} >= -c_i と b_{i,j} <= c_i に分解
 *
 * 目的関数:
 *   minimize Σ c_i
 *
 * コンパイル例:  gcc enclosing_zonotope_2d.c -lglpk -o ezono
 * 実行例:       ./ezono
 */

#include <stdio.h>
#include <stdlib.h>
#include <glpk.h>

/* 2次元の点やベクトルを表す構造体 */
typedef struct {
    double x;
    double y;
} Vec2;

/*
 * solve_enclosing_zonotope_2d():
 *   点集合 points[0..n-1], 単位ベクトル集合 vectors[0..k-1] を受け取り，
 *   GLPK で LP を解いて，
 *     p = (p_x, p_y),
 *     c[0..k-1],
 *     b[(i,j)]  (i=0..k-1, j=0..n-1)
 *   を求める。
 *
 *   戻り値は 0:最適解を得た, それ以外:エラー。
 */
int solve_enclosing_zonotope_2d(const Vec2 *points, int n,
                                const Vec2 *vectors, int k)
{
    /* 変数数:
     *   1) p_x
     *   2) p_y
     *   3) c_i が k個
     *   4) b_{i,j} が k*n 個
     * 合計 = 2 + k + k*n
     */
    int numCols = 2 + k + (k * n);

    /* 制約数:
     *   1) 各点に対して x座標の等式, y座標の等式 → 2*n
     *   2) -c_i <= b_{i,j} <= c_i → これを2つの不等式に分解 → 2 * k * n
     * 合計 = 2*n + 2*k*n
     */
    int numRows = 2 * n + 2 * k * n;

    /* GLPK問題オブジェクトの作成 */
    glp_prob *lp = glp_create_prob();
    glp_set_prob_name(lp, "Enclosing_Zonotope_2D");
    glp_set_obj_dir(lp, GLP_MIN);

    /* 行と列を作成 (1-based index) */
    glp_add_rows(lp, numRows);
    glp_add_cols(lp, numCols);

    /* --- 変数カラムに名前, 下限/上限, 目的関数係数などを設定 --- */

    /* 順番を決める:
     *  col 1  = p_x
     *  col 2  = p_y
     *  col(3..(2+k-1)) = c1..c_k (c_i >= 0)
     *  col(3+k..(2+k+k*n-1)) = b_{i,j}
     *
     * ただし注: i,j は0-based なので実際に index 計算して割り振る
     */

    /* ヘルパー: b_{i,j} のカラムインデックスを求める関数 */
    int colIndex_b(int i, int j, int k, int n) {
        /* b_{i,j} の開始は colStart = 2 + k + 1 (1-based) */
        int colStart = 3 + k; /* 1-based index for the first b_{i,j} */
        /* i行 j列目 → i*n + j のオフセット */
        return colStart + (i * n + j);
    }

    /* p_x, p_y (制限なし, 目的関数係数=0) */
    glp_set_col_name(lp, 1, "p_x");
    glp_set_col_bnds(lp, 1, GLP_FR, 0.0, 0.0); /* FR = free (no bounds) */
    glp_set_obj_coef(lp, 1, 0.0);

    glp_set_col_name(lp, 2, "p_y");
    glp_set_col_bnds(lp, 2, GLP_FR, 0.0, 0.0);
    glp_set_obj_coef(lp, 2, 0.0);

    /* c_i >= 0, 目的関数は Σ c_i */
    for(int i = 0; i < k; i++){
        int colID = 3 + i;
        char cname[32];
        sprintf(cname, "c_%d", i);
        glp_set_col_name(lp, colID, cname);
        glp_set_col_bnds(lp, colID, GLP_LO, 0.0, 0.0); /* 下限0, 上限なし */
        glp_set_obj_coef(lp, colID, 1.0); /* 目的関数に1で加算 */
    }

    /* b_{i,j}, -∞~+∞, 目的関数係数=0 */
    for(int i = 0; i < k; i++){
        for(int j = 0; j < n; j++){
            int colID = colIndex_b(i, j, k, n);
            char bname[32];
            sprintf(bname, "b_%d_%d", i, j);
            glp_set_col_name(lp, colID, bname);
            glp_set_col_bnds(lp, colID, GLP_FR, 0.0, 0.0); /* 制限は後で行制約で付ける */
            glp_set_obj_coef(lp, colID, 0.0);
        }
    }

    /*
     * --- 行制約の設定 ---
     *   合計(2*n + 2*k*n)個の行を作り，
     *   それぞれに係数を割り当てる。
     *
     *   GLPK では「各行に対して，その行内で非ゼロのカラム数ぶん
     *    ia[row_i, col_j], ar[...] という配列にセットする」形式。
     *
     *   ここでは全てまとめて1つの配列に詰める方法でいきます。
     *   non-zero 係数の総数を計算して，glp_load_matrix() で一度に設定する。
     */

    /* (1) 各点 p_j の x座標制約: x_j = p_x + Σ b_{i,j} * vx_i
     *     → 左辺 - 右辺 = 0  に変形すると
     *        p_x + Σ(b_{i,j} * vx_i) - x_j = 0
     *        ただし x_j は定数なので  rowBound = (0,0)
     *
     *     同様に y座標制約: p_y + Σ(b_{i,j} * vy_i) - y_j = 0
     *
     *     合計 2*n 本の等式制約
     *
     * (2) -c_i <= b_{i,j} <= c_i
     *     2つの不等式:
     *       b_{i,j} + c_i >= 0
     *       c_i - b_{i,j} >= 0
     *     合計 2*k*n 本
     */

    int totalNonZero = 0;
    /* (1) 各点の x,y 座標制約での非ゼロ要素数を数える */
    /*     x座標: p_x,  b_{i,j} * vx_i → k個
     *     y座標: p_y,  b_{i,j} * vy_i → k個
     *     これが n点分ある → 2*n*(1 + k) 個
     */
    totalNonZero += 2 * n * (1 + k);

    /* (2) b_{i,j} + c_i >= 0 と c_i - b_{i,j} >= 0
     *     各行について 非ゼロ係数は2つ (b_{i,j} と c_i)
     *     これが2行/組 × k*n個 → 2 * k*n * 2 = 4*k*n
     */
    totalNonZero += 4 * k * n;

    /* 配列を確保 (GLPKは1-based indexなので+1しておく) */
    int    *ia = (int*) malloc((totalNonZero+1) * sizeof(int));
    int    *ja = (int*) malloc((totalNonZero+1) * sizeof(int));
    double *ar = (double*) malloc((totalNonZero+1) * sizeof(double));

    int rowCounter = 0;      /* どの行を埋めているか(1-based) */
    int nnzCounter = 0;      /* 非ゼロ係数を埋める位置(1-based) */

    /* ----- (1) x,y座標の等式制約 (2*n本) ----- */

    for(int j = 0; j < n; j++){
        /* x座標: row = ++rowCounter */
        rowCounter++;
        double xj = points[j].x;
        /* p_x + Σ(b_{i,j} * vx_i) = x_j
         * → p_x + Σ(...) - x_j = 0
         *
         * rowの両辺 = 0 としたい → GLP_FX (fixed) で(0,0)
         */
        {
            char rname[32];
            sprintf(rname, "constraint_x_%d", j);
            glp_set_row_name(lp, rowCounter, rname);
            glp_set_row_bnds(lp, rowCounter, GLP_FX, 0.0, 0.0);

            /* p_x の係数 +1 */
            nnzCounter++;
            ia[nnzCounter] = rowCounter;
            ja[nnzCounter] = 1; /* col=1 -> p_x */
            ar[nnzCounter] = 1.0;

            /* b_{i,j} の係数 vx_i */
            for(int i = 0; i < k; i++){
                nnzCounter++;
                ia[nnzCounter] = rowCounter;
                ja[nnzCounter] = colIndex_b(i, j, k, n);
                ar[nnzCounter] = vectors[i].x;
            }

            /* 定数 -x_j → GLPKの形式では行の左右辺に持ち込めないので
             *  "row bounds" で 0 に固定しつつ，
             *   係数としては書かない手法が一般的。
             *   あるいは p_x + sum(...) - x_j = 0 の "- x_j" は
             *   row bound で + x_j として扱う方法もあるが，
             *   ここでは "∑ var - x_j = 0" → row bound(0,0) にしておけばOK。
             *   x_j は定数だから GLPK で係数としては不要です。
             */
        }

        /* y座標: row = ++rowCounter */
        rowCounter++;
        double yj = points[j].y;
        {
            char rname[32];
            sprintf(rname, "constraint_y_%d", j);
            glp_set_row_name(lp, rowCounter, rname);
            glp_set_row_bnds(lp, rowCounter, GLP_FX, 0.0, 0.0);

            /* p_y の係数 +1 */
            nnzCounter++;
            ia[nnzCounter] = rowCounter;
            ja[nnzCounter] = 2; /* col=2 -> p_y */
            ar[nnzCounter] = 1.0;

            /* b_{i,j} の係数 vy_i */
            for(int i = 0; i < k; i++){
                nnzCounter++;
                ia[nnzCounter] = rowCounter;
                ja[nnzCounter] = colIndex_b(i, j, k, n);
                ar[nnzCounter] = vectors[i].y;
            }
        }
    }

    /* ----- (2) -c_i <= b_{i,j} <= c_i  → 2つの不等式に分解 ----- */
    for(int i = 0; i < k; i++){
        for(int j = 0; j < n; j++){
            /* b_{i,j} + c_i >= 0 */
            rowCounter++;
            {
                char rname[32];
                sprintf(rname, "bound_lo_%d_%d", i, j);
                glp_set_row_name(lp, rowCounter, rname);
                /*
                 * b_{i,j} + c_i >= 0 → rowは下限0,上限∞
                 */
                glp_set_row_bnds(lp, rowCounter, GLP_LO, 0.0, 0.0);

                /* b_{i,j} の係数 +1 */
                nnzCounter++;
                ia[nnzCounter] = rowCounter;
                ja[nnzCounter] = colIndex_b(i, j, k, n);
                ar[nnzCounter] = 1.0;

                /* c_i の係数 +1 */
                nnzCounter++;
                ia[nnzCounter] = rowCounter;
                ja[nnzCounter] = 3 + i; /* c_i */
                ar[nnzCounter] = 1.0;
            }

            /* c_i - b_{i,j} >= 0 */
            rowCounter++;
            {
                char rname[32];
                sprintf(rname, "bound_up_%d_%d", i, j);
                glp_set_row_name(lp, rowCounter, rname);
                /* c_i - b_{i,j} >= 0 → row下限0,上限∞ */
                glp_set_row_bnds(lp, rowCounter, GLP_LO, 0.0, 0.0);

                /* c_i の係数 +1 */
                nnzCounter++;
                ia[nnzCounter] = rowCounter;
                ja[nnzCounter] = 3 + i;
                ar[nnzCounter] = 1.0;

                /* b_{i,j} の係数 -1 */
                nnzCounter++;
                ia[nnzCounter] = rowCounter;
                ja[nnzCounter] = colIndex_b(i, j, k, n);
                ar[nnzCounter] = -1.0;
            }
        }
    }

    /* すべての行に対する係数を glp_load_matrix() で一括設定 */
    glp_load_matrix(lp, nnzCounter, ia, ja, ar);

    /* 単純形法を実行 */
    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF; /* 計算ログを消したい場合 */
    int ret = glp_simplex(lp, &parm);
    if(ret != 0){
        fprintf(stderr, "Error: glp_simplex failed (ret=%d)\n", ret);
        glp_delete_prob(lp);
        free(ia); free(ja); free(ar);
        return -1;
    }

    /* 解の取得 */
    int status = glp_get_status(lp);
    if(status != GLP_OPT){
        fprintf(stderr, "No optimal solution found (status=%d)\n", status);
        glp_delete_prob(lp);
        free(ia); free(ja); free(ar);
        return -1;
    }

    double z = glp_get_obj_val(lp);
    printf("Optimal objective value: %.6f\n", z);

    /* p_x, p_y */
    double px_val = glp_get_col_prim(lp, 1);
    double py_val = glp_get_col_prim(lp, 2);
    printf("p = (%.6f, %.6f)\n", px_val, py_val);

    /* c_i */
    for(int i = 0; i < k; i++){
        int colID = 3 + i;
        double c_val = glp_get_col_prim(lp, colID);
        printf("c_%d = %.6f\n", i, c_val);
    }

    /* b_{i,j} (必要なら表示する)
     * for(int i = 0; i < k; i++){
     *     for(int j = 0; j < n; j++){
     *         int colID = colIndex_b(i, j, k, n);
     *         double b_val = glp_get_col_prim(lp, colID);
     *         printf("b_{%d,%d} = %.6f\n", i, j, b_val);
     *     }
     * }
     */

    /* 後始末 */
    glp_delete_prob(lp);
    free(ia);
    free(ja);
    free(ar);

    return 0; /* success */
}

/* ===== メイン関数 ===== */
int main(void)
{
    /*
     * 例として、点3つ (n=3) と単位ベクトル2つ (k=2) を与える。
     * Python版サンプルと同様、 x方向と y方向の2つの単位ベクトルを使う。
     */
    Vec2 points[3] = {
        {0.0, 2.0},
        {3.0, 2.0},
        {1.0, 0.0}
    };
    int n = 3;

    /* 単位ベクトル2つ */
    Vec2 vectors[2] = {
        {1.0, 0.0}, /* x方向 */
        {0.0, 1.0}  /* y方向 */
    };
    int k = 2;

    int ret = solve_enclosing_zonotope_2d(points, n, vectors, k);
    if(ret == 0){
        printf("Solve done.\n");
    } else {
        printf("Solve failed.\n");
    }

    return 0;
}
