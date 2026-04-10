try:
    a = int(input("Enter a number: "))
    print(1/a)

except ValueError:
    print("Only number")

except Exception as e:
    print(f"{e} occured")

finally:
    b = 34

print("hello world")
print(b)