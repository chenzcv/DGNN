IP_Table=[58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
	62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
	57, 49, 41, 33, 25, 17,  9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
	61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7]

PC1 = [57, 49, 41, 33, 25, 17, 9,
        1,	58,	50,	42,	34,	26,	18,
        10,	2,	59,	51,	43,	35,	27,
        19,	11,	3,	60,	52,	44,	36,
        63,	55,	47,	39,	31,	23,	15,
        7,	62,	54,	46,	38,	30,	22,
        14,	6,	61,	53,	45,	37,	29,
        21,	13,	5,	28,	20,	12,	4]

PC2 = [14,	17,	11,	24,	1,	5,
        3,	28,	15,	6,	21,	10,
        23,	19,	12,	4,	26,	8,
        16,	7,	27,	20,	13,	2,
        41,	52,	31,	37,	47,	55,
        30,	40,	51,	45,	33,	48,
        44,	49,	39,	56,	34,	53,
        46,	42,	50,	36,	29,	32]

EP = [32, 1, 2,	3,	4,	5,
    4,	5,	6,	7,	8,	9,
    8,	9,	10,	11,	12,	13,
    12,	13,	14,	15,	16,	17,
    16,	17,	18,	19,	20,	21,
    20,	21,	22,	23,	24,	25,
    24,	25,	26,	27,	28,	29,
    28,	29,	30,	31,	32,	1]

P_box = [16,7,20,21,29,12,28,17,
         1, 15,23,26,5,18,31,10,
         2, 8, 24,14,32,27,3,9,
         19,13,30,6,22,11,4,25]

src=[1,0,1,0,0,1,1,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,0, 0,0,1,1, 1,1,1,1,
     1,1,0,1, 0,1,0,0, 1,1,0,0, 1,0,0,0, 1,0,0,1, 0,1,1,0, 1,1,0,1, 0,0,1,0]

dst=[0 for _ in range(64)]
dst_key=[0 for _ in range(56)]
dst_key2=[0 for _ in range(48)]

key='0001101001011101011011011000100101011011010010110110011011011011'
key=list(key)
print(len(key))
for i in range(64):
    dst[i] = src[IP_Table[i] - 1]
result = ''.join([str(x) for x in dst])
print(result)

for i in range(56):
    dst_key[i] = key[PC1[i]-1]
new_key = ''.join([str(x) for x in dst_key])
print(new_key)

new_key = '00010001111011001000100100111110001010001101011111100111'
# new_key = '11100001100110010101010111111010101011001100111100011110'
key=list(new_key)
for i in range(48):
    dst_key2[i] = key[PC2[i]-1]
new_key = ''.join([str(x) for x in dst_key2])
print(new_key)

R0='11110111000011110010110011001011'
# R0='00000000111111110000011010000011'
R0=list(R0)
R=[0 for _ in range(48)]
for i in range(48):
    R[i] = R0[EP[i]-1]
R_str = ''.join([str(x) for x in R])
print(R_str)

R_K=[0 for _ in range(48)]
for i in range(48):
    if dst_key2[i] == R[i]:
        R_K[i]='0'
    else:
        R_K[i]='1'
R_K_str = ''.join([str(x) for x in R_K])
print(R_K_str)

R_S='01001110010010001010011110101000'
R_S=list(R_S)
R_P=[0 for _ in range(32)]
for i in range(32):
    R_P[i] = R_S[P_box[i] - 1]
R_P_str = ''.join([str(x) for x in R_P])
print(R_P_str)

R_L=[0 for _ in range(32)]
L0='10110010110111000101110100001001'
L0=list(L0)
print(len(L0))
for i in range(32):
    if L0[i] == R_P[i]:
        R_L[i]='0'
    else:
        R_L[i]='1'
R_L_str = ''.join([str(x) for x in R_L])
print(R_L_str)


# Caesar Cipher Hacker
# https://www.nostarch.com/crackingcodes (BSD Licensed)

# message = 'guv6Jv6Jz!J6rp5r7Jzr66ntrM'
message='qeFIP?eGSeECNNS,>5coOMXXcoPSZIWoQI,>avnl1olyD4l\'ylDohww6DhzDjhuDil,>z.GM?.cEQc. 70c.7KcKMKHA9AGFK,>?MFYp2pPJJUpZSIJWpRdpMFY,>ZqH8sl5HtqHTH4s3lyvH5zH5spH4t pHzqHlH3l5K>Zfbi,!tif!xpvme!qspcbcmz!fbu!nfA'
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.'
message_list = message.split('>')
# Loop through every possible key:
for message in message_list:
    for key in range(len(SYMBOLS)):
        # It is important to set translated to the blank string so that the
        # previous iteration's value for translated is cleared.
        translated = ''

        # The rest of the program is almost the same as the original program:

        # Loop through each symbol in `message`:
        for symbol in message:
            if symbol in SYMBOLS:
                symbolIndex = SYMBOLS.find(symbol)
                translatedIndex = symbolIndex - key

                # Handle the wrap-around:
                if translatedIndex < 0:
                    translatedIndex = translatedIndex + len(SYMBOLS)

                # Append the decrypted symbol:
                translated = translated + SYMBOLS[translatedIndex]

            else:
                # Append the symbol without encrypting/decrypting:
                translated = translated + symbol

        # Display every possible decryption:
        print('Key #%s: %s' % (key, translated))
    print('-----------------------------------------------------------------------------')
