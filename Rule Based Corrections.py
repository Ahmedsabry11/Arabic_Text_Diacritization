def primary_diacritics_corrections(predicted_diacritized_string):
    ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
    ARABIC_LETTERS = ARABIC_LETTERS.union({' '})
    corrected_string = list(predicted_diacritized_string)
    print(corrected_string)
    i=0
    while i < len(corrected_string):
        if corrected_string[i] == 'إ':
            while i+1 < len(corrected_string) and corrected_string[i+1] not in ARABIC_LETTERS:
                corrected_string.pop(i+1)
            corrected_string.insert(i+1, 'ِ')
            # print(corrected_string)
            # print("here1")
        
        if corrected_string[i] in ['ى','ة']:
            while corrected_string[i-1] not in ARABIC_LETTERS:
                corrected_string.pop(i-1)
            corrected_string.insert(i, 'َ')
            i+=1
            # print(corrected_string)
            # print("here2")

        if corrected_string[i] == 'ا' and not((corrected_string[i+1]==' ') or (corrected_string[i+1] == 'َ' and corrected_string[i+2]==' ') or (corrected_string[i+1] == 'ً' and corrected_string[i+2]==' ')) :
            while corrected_string[i+1] not in ARABIC_LETTERS:
                i+=1
            if corrected_string[i+1] == ' ':
                corrected_string.insert(i+1,'ً')
            # print(corrected_string)
            # print("here3")

        if corrected_string[i] == 'ا':
            print(corrected_string)
            while corrected_string[i+1] not in ARABIC_LETTERS:
                i+=1
            if corrected_string[i+1] != ' ':
                while corrected_string[i-1] not in ARABIC_LETTERS:
                    corrected_string.pop(i-1)
                corrected_string.insert(i, 'َ')
                i+=1
            # print(corrected_string)
            # print("here4")
        elif i+1 < len(corrected_string) and corrected_string[i] in [' ','ى','آ','ا']:
            while corrected_string[i+1] not in ARABIC_LETTERS:
                corrected_string.pop(i+1)
            # print(corrected_string)
            # print("here8")

            
        if i+1 < len(corrected_string) and corrected_string[i+1] == 'ْ' and (corrected_string[i-1] == ' ' or i==0):
            corrected_string.pop(i+1)
            # print(corrected_string)
            # print("here5")

        if corrected_string[i] in ['ً','ٍ','ٌ'] and corrected_string[i+1] != ' ':
            corrected_string.pop(i)
            # print(corrected_string)
            # print("here6")

        if i+1 < len(corrected_string) and corrected_string[i] not in ['ء','ة','ا'] and corrected_string[i+1] == 'ً' and  corrected_string[i+2] == ' ':
            corrected_string.pop(i+1)
            # print(corrected_string)
            # print("here7")

        i+=1
    print(corrected_string)
    return ''.join(corrected_string)
    

predicted_string = 'بى بة'
corrected_result = primary_diacritics_corrections(predicted_string)
print(corrected_result)


def Shadda_Corrections(predicted_diacritized_string):
    forbidden_char = [' ','ا','أ','آ','ى','ئ','ء','إ','ة']
    ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
    ARABIC_LETTERS = ARABIC_LETTERS.union({' '})
    corrected_string = list(predicted_diacritized_string)
    i=0
    while i < len(corrected_string):
        char = corrected_string[i]
        if char in forbidden_char or i==0 or corrected_string[i-1]==" ":
            if i + 1 < len(corrected_string) and corrected_string[i+1] == 'ّ':
                corrected_string.pop(i+1)
                if i + 2 < len(corrected_string) and corrected_string[i+2] not in ARABIC_LETTERS:
                    corrected_string.pop(i+1)

            elif i + 2 < len(corrected_string) and corrected_string[i+2] == 'ّ' and corrected_string[i+1] not in ARABIC_LETTERS:
                corrected_string.pop(i+1)
                corrected_string.pop(i+1)
        i += 1
    return ''.join(corrected_string)


