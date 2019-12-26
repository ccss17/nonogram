def division(num, limit=float('inf')):
    for main_chunk in range(min(num, limit), 0, -1):
        rest_chunk = num - main_chunk
        rest_chunk_divisions = tuple(division(rest_chunk, limit=main_chunk))
        if rest_chunk_divisions:
            for rest_chunk_division in rest_chunk_divisions:
                yield main_chunk, *rest_chunk_division
        else: 
            yield main_chunk,

for i in range(8):
    print('='*20, 'TEST for', i, '='*20)
    for v in division(i):
        print(v)
