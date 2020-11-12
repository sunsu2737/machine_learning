

f = open('cars.csv', 'r')   # 파일을 open한다.
head = f.readline()           # 파일의 첫 라인을 읽는다. 첫 라인은 테이블의 head이다.

data = f.readlines()          # 파일의 나머지 모든 라인을 읽어온다. 라인들의 리스트로 저장된다.
f.close()

cnt = 0
bmw_list = [list(car.strip().split(',')) for car in data if 'BMW' in car]
for car in data:
    if 'Volkswagen' in car and 'sedan' in car:
        cnt += 1

bmw_list.sort(key=lambda x: x[7])
print('실습2-1:', cnt)

print(bmw_list)
