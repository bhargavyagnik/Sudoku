import image
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]

rows = 'ABCDEFGHI'
cols = '123456789'
positions=cross(rows,cols)
BOARD={key:None for key in positions}
scan=image.getval()

