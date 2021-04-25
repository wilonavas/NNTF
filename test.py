str1 = "PCIE Slot 7"
str2 = "Slot 71"


for x in str1.split():
    if x.isdigit() and (x in str2.split()):
        print(f'match:{x}')
    else:
        print(f'No Match: {x}')

