#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ------------------------------------
 *  Vec2: 2次元ベクトル（x, y）
 * ------------------------------------ */
typedef struct {
    double x;
    double y;
} Vec2;

/* ------------------------------------
 *  動的配列を扱うための簡易コンテナ
 * ------------------------------------ */
typedef struct {
    Vec2 *data;
    size_t size;
    size_t capacity;
} Vec2Array;

/* ------------------------------------
 *  Vec2Array の初期化
 * ------------------------------------ */
void initVec2Array(Vec2Array *arr, size_t initCapacity) {
    arr->data = (Vec2 *)malloc(initCapacity * sizeof(Vec2));
    arr->size = 0;
    arr->capacity = initCapacity;
}

/* ------------------------------------
 *  Vec2Array のメモリ解放
 * ------------------------------------ */
void freeVec2Array(Vec2Array *arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}

/* ------------------------------------
 *  Vec2Array に要素を追加
 * ------------------------------------ */
void pushBackVec2Array(Vec2Array *arr, Vec2 v) {
    if (arr->size == arr->capacity) {
        size_t newCapacity = arr->capacity * 2;
        Vec2 *newData = (Vec2 *)realloc(arr->data, newCapacity * sizeof(Vec2));
        if (!newData) {
            fprintf(stderr, "Error: Memory reallocation failed\n");
            exit(EXIT_FAILURE);
        }
        arr->data = newData;
        arr->capacity = newCapacity;
    }
    arr->data[arr->size++] = v;
}

/* ------------------------------------
 *  ベクトルの加算
 * ------------------------------------ */
static inline Vec2 addVec2(const Vec2 a, const Vec2 b) {
    Vec2 r = {a.x + b.x, a.y + b.y};
    return r;
}

/* ------------------------------------
 *  2D クロス積の大きさ (ベクトル a から b への外積 z 成分)
 *  ※凸包などのソート判定に使う
 * ------------------------------------ */
static inline double cross(const Vec2 a, const Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

/* ------------------------------------
 *  点列を x 昇順, y 昇順でソート
 * ------------------------------------ */
static int compareVec2(const void *p1, const void *p2) {
    Vec2 *v1 = (Vec2 *)p1;
    Vec2 *v2 = (Vec2 *)p2;
    if (v1->x < v2->x) return -1;
    if (v1->x > v2->x) return 1;
    if (v1->y < v2->y) return -1;
    if (v1->y > v2->y) return 1;
    return 0;
}

/* ------------------------------------
 *  凸包（Andrew's Monotone Chain）を求める関数
 *  与えられた点集合 points から凸包を作り、result に格納
 * ------------------------------------ */
void computeConvexHull(const Vec2Array *points, Vec2Array *result) {
    // 1. ソート
    qsort(points->data, points->size, sizeof(Vec2), compareVec2);

    // 2. 下側のチェーンを構築
    for (size_t i = 0; i < points->size; i++) {
        while (result->size >= 2) {
            Vec2 a = result->data[result->size - 2];
            Vec2 b = result->data[result->size - 1];
            Vec2 c = points->data[i];
            Vec2 ab = {b.x - a.x, b.y - a.y};
            Vec2 ac = {c.x - a.x, c.y - a.y};
            if (cross(ab, ac) <= 0.0) {
                // 左回りでなければ pop
                result->size--;
            } else {
                break;
            }
        }
        pushBackVec2Array(result, points->data[i]);
    }

    // 3. 上側のチェーンを構築
    size_t lowerSize = result->size;
    for (int i = (int)points->size - 2, t = (int)result->size + 1; i >= 0; i--) {
        while (result->size >= lowerSize + 1) {
            Vec2 a = result->data[result->size - 2];
            Vec2 b = result->data[result->size - 1];
            Vec2 c = points->data[i];
            Vec2 ab = {b.x - a.x, b.y - a.y};
            Vec2 ac = {c.x - a.x, c.y - a.y};
            if (cross(ab, ac) <= 0.0) {
                // 左回りでなければ pop
                result->size--;
            } else {
                break;
            }
        }
        pushBackVec2Array(result, points->data[i]);
    }

    // 4. 重複した最後の頂点を削除
    result->size--;
}

/* ------------------------------------
 *  単純化したミンコフスキー和 (A ⊕ B)
 *  - A と B は多角形を想定（頂点集合）
 *  - 全組み合わせ (a + b) を取ってから凸包を作る形
 * ------------------------------------ */
void minkowskiSum(const Vec2Array *A, const Vec2Array *B, Vec2Array *out) {
    // (a + b) を全て列挙
    for (size_t i = 0; i < A->size; i++) {
        for (size_t j = 0; j < B->size; j++) {
            pushBackVec2Array(out, addVec2(A->data[i], B->data[j]));
        }
    }
    // 列挙した点の凸包を作り、out に再度格納
    Vec2Array hull;
    initVec2Array(&hull, out->size * 2);
    computeConvexHull(out, &hull);

    // out を hull の内容で上書き
    free(out->data);
    out->data = hull.data;
    out->size = hull.size;
    out->capacity = hull.capacity;
}

/* ------------------------------------
 *  線分の集合を順番にミンコフスキー和して
 *  最終的なゾノトープを得る関数
 * ------------------------------------ */
void buildZonotope(Vec2 *generators, size_t numGenerators, Vec2Array *result) {
    // 初期形状は原点1点のみ
    initVec2Array(result, 4);
    Vec2 origin = {0.0, 0.0};
    pushBackVec2Array(result, origin);

    // 各線分を一つずつ足していく
    for (size_t i = 0; i < numGenerators; i++) {
        // 線分は「± generator[i]」で構成される2点
        Vec2Array seg;
        initVec2Array(&seg, 2);

        Vec2 p1 = {+generators[i].x, +generators[i].y};
        Vec2 p2 = {-generators[i].x, -generators[i].y};

        pushBackVec2Array(&seg, p1);
        pushBackVec2Array(&seg, p2);

        // 今ある図形 result と seg のミンコフスキー和を取る
        Vec2Array newShape;
        initVec2Array(&newShape, result->size * seg.size * 2);

        minkowskiSum(result, &seg, &newShape);

        // result を更新
        freeVec2Array(result);  // 中身は newShape に移動したので消してOK
        *result = newShape;

        freeVec2Array(&seg);
    }
}

/* ------------------------------------
 *  メイン関数
 * ------------------------------------ */
int main(void) {
    // 例として、3本のジェネレーターベクトルを与える
    // ここを好きなベクトル集合に差し替えてみてください。
    Vec2 generators[] = {
        {2.0, 1.0},
        {0.0, 3.0},
        {1.0, -2.0},
        {1.0,0.5},
        {0.1,-0.5}
    };
    size_t numGenerators = sizeof(generators) / sizeof(generators[0]);

    // ゾノトープを作る
    Vec2Array zonotope;
    buildZonotope(generators, numGenerators, &zonotope);

    // 結果の頂点を出力
    printf("Zonotope vertices (in convex hull order):\n");
    for (size_t i = 0; i < zonotope.size; i++) {
        printf("  (%.3f, %.3f)\n", zonotope.data[i].x, zonotope.data[i].y);
    }

    // 後始末
    freeVec2Array(&zonotope);
    return 0;
}
