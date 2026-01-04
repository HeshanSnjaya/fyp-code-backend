from app.ml_classifier import classifier

classifier.load_model()

test_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

label, confidence = classifier.predict(test_code)
print(f"Predicted: {label} ({confidence:.2f})")
