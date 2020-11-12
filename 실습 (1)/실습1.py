def insertion_sort(data):
    sorted_data = []
    for i in data:
        if sorted_data:
            for j in range(len(sorted_data)):
                if i < sorted_data[j]:
                    sorted_data.insert(j, i)
                    break
                if j == len(sorted_data)-1:
                    sorted_data.append(i)
        else:
            sorted_data.append(i)
    return sorted_data


def divide(data):
    return data[:len(data)//2], data[len(data)//2:]


def merge(left, right):
    result = []
    i, j = 0, 0

    while i < len(left) and j < len(right):
        if left[i] > right[j]:
            result.append(right[j])
            j += 1
        else:
            result.append(left[i])
            i += 1
    if i == len(left):
        for k in range(j, len(right)):
            result.append(right[k])
    elif j == len(right):
        for k in range(i, len(left)):
            result.append(left[k])
    return result


def merge_sort(data):

    if len(data) <= 1:
        return data
    l, r = divide(data)
    left, right = merge_sort(l), merge_sort(r)
    return merge(left, right)


data = [2, 1, 5, 41, 8, 8, 8, 623, 7]

print(merge_sort(data))
data = [2, 1, 5, 41, 8, 8, 8, 623, 7]
print(insertion_sort(data))
