

f = open('cars.csv', 'r')   # 파일을 open한다.
# 파일의 첫 라인을 읽는다. 첫 라인은 테이블의 head이다.
head = list(f.readline().strip().split(','))
# Brand,Price,Body,Mileage,EngineV,Engine Type,Registration,Year,Model
data = []
while True:
    car = f.readline().strip().split(',')

    if car == ['']:
        break
    data.append({head[i]: car[i] for i in range(len(car))})
f.close()

problem2 = sorted([car for car in data if car['Price'] != 'NA' and 20000 <= float(car['Price']) <= 50000 and 2000 < int(car['Year'])
            and car['Body'] == 'sedan' and car['Engine Type'] == 'Gas'],key=lambda x: x['Price'])
for i in problem2:
    print(i)

problem3 = {car['Brand'] for car in data}

print(problem3)