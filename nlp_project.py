def Shadda_Corrections(predicted_diacritized_string):
    forbidden_char = ['0',' ','ا','أ','آ','ى','ئ','ء','إ','ة']
    ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
    corrected_string = list(predicted_diacritized_string)
    print(corrected_string)
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
    print(corrected_string)
    return ''.join(corrected_string)