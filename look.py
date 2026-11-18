from sklearn.datasets import load_digits

# 1. get the free digit pictures
digits = load_digits()

print("How many pictures:", len(digits.images))   # 1797

# 2. take the first picture and its correct answer
picture = digits.images[4]     # an 8×8 grid of numbers
answer = digits.target[4]      # the real digit (0–9)

print("This picture is a:", answer)

# 3. show the 64 numbers (0 = white, 16 = black)
print(picture)

# 4. draw it with characters so we can SEE it
for row in picture:
    line = ""
    for dot in row:
        if dot > 8:
            line += "██"   # dark square
        elif dot > 0:
            line += "░░"   # light square
        else:
            line += "  "   # white square
    print(line)
