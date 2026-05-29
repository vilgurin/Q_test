def solve(m, n, arr):
    def dfs(ri, ci):
        if ci < n - 1 and arr[ri][ci + 1] == 1:
            arr[ri][ci + 1] = 0
            dfs(ri, ci + 1)
        if ci > 0 and arr[ri][ci - 1] == 1:
            arr[ri][ci - 1] = 0
            dfs(ri, ci - 1)
        if ri < m - 1 and arr[ri + 1][ci] == 1:
            arr[ri + 1][ci] = 0
            dfs(ri + 1, ci)
        if ri > 0 and arr[ri - 1][ci] == 1:
            arr[ri - 1][ci] = 0
            dfs(ri - 1, ci)


    count = 0

    for ri, row in enumerate(arr):
        for ci, cell in enumerate(row):
            if cell == 1:
                count += 1
                arr[ri][ci] = 0
                dfs(ri, ci)

    return count

if __name__ == "__main__":
    m, n = map(int, input().split())
    arr = []
    for _ in range(m):
        arr.append([int(x) for x in input().split()])

    print(solve(m, n, arr))