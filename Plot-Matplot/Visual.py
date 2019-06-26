import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def average(Matrix):
    sum_value = np.sum(Matrix)
    average = sum_value /(19*19)
    return average


def one_layer():
    a =np.load("sentence_embeddingL1.npy")
    a = a[0][0] + a[0][1]
    #matrix = a.reshape((2,256,-1))[0][0:19]
    matrix = a[0:19]
    similiary_matrix = []

    for vector1 in matrix:
        for vector2 in matrix:
            entries = (np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))))
            similiary_matrix.append(entries)

    similiary_matrix = np.array(similiary_matrix).reshape(19,19)
    l1=average(similiary_matrix)
    print("l1={}".format(l1))
    return similiary_matrix

def two_layer():
    a =np.load("sentence_embedding512.npy")
    #a = (a[0][0]+a[0][1])
    a = a[0][0]
    matrix = a[0:19]
    similiary_matrix = []

    for vector1 in matrix:
        for vector2 in matrix:
            #entries=np.sqrt(np.sum(np.square(vector1-vector2)))
            entries = (np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))))
            similiary_matrix.append(entries)

    similiary_matrix = np.array(similiary_matrix).reshape(19,19)
    l2=average(similiary_matrix)
    print("l2={}".format(l2))
    # similiary_matrix[4][17]=similiary_matrix[17][4]=0.8
    return similiary_matrix
def three_layer():
    a =np.load("sentence_embeddingL3.npy")
    a = a[0][0]*a[0][1]
    #matrix = a.reshape((2,256,-1))[0][0:19]
    matrix = a[0:19]
    similiary_matrix = []

    for vector1 in matrix:
        for vector2 in matrix:
            #entries=np.sqrt(np.sum(np.square(vector1-vector2)))
            entries = (np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))))
            similiary_matrix.append(entries)

    similiary_matrix = np.array(similiary_matrix).reshape(19,19)
    l3=average(similiary_matrix)
    print("l3={}".format(l3))
    return similiary_matrix

def hidden_216():
    a =np.load("sentence_embeddingL1.npy")
    a = a[0][0] + a[0][1]
    #matrix = a.reshape((2,256,-1))[0][0:19]
    matrix = a[0:19]
    similiary_matrix = []

    for vector1 in matrix:
        for vector2 in matrix:
            entries = (np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))))
            similiary_matrix.append(entries)

    similiary_matrix = np.array(similiary_matrix).reshape(19,19)
    h1=average(similiary_matrix)
    print("h1={}".format(h1))
    return similiary_matrix
def hidden_1024():
    a =np.load("sentence_embeddingL2.npy")
    a = a[0][0] + a[0][1]
    #matrix = a.reshape((2,256,-1))[0][0:19]
    matrix = a[0:19]
    similiary_matrix = []

    for vector1 in matrix:
        for vector2 in matrix:
            #entries=np.sqrt(np.sum(np.square(vector1-vector2)))
            entries = (np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))))
            similiary_matrix.append(entries)

    similiary_matrix = np.array(similiary_matrix).reshape(19,19)
    h2=average(similiary_matrix)
    print("h3={}".format(h2))
    return similiary_matrix
def hidden_512():
    a =np.load("sentence_embedding.txt.npy")
    # print(a.shape)
    a = a[0][0]+a[0][1]
    # a = a[0][1]
    # matrix = a[19:38]
    matrix = a[197:216]
    similiary_matrix = []

    for vector1 in matrix:
        for vector2 in matrix:
            #entries=np.sqrt(np.sum(np.square(vector1-vector2)))
            entries = (np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))))
            similiary_matrix.append(entries)

    similiary_matrix = np.array(similiary_matrix).reshape(19,19)
    h3=average(similiary_matrix)
    print("h2={}".format(h3))
    # similiary_matrix[4][17]=similiary_matrix[17][4]=0.8
    return similiary_matrix

fig, ([ax,ax2,ax3],[ax4,ax5,ax6])= plt.subplots(nrows=2, ncols=3,figsize=(8,8))


vegetables = ["Who","made", "these", "wonderful", "cars", "that", "people","drove","as","if", "they", "were","an", "extension", "of", "their", "own","bodies", "?"]
farmers = ["Who","made", "these", "wonderful", "cars", "that", "people","drove","as","if", "they", "were","an", "extension", "of", "their", "own","bodies", "?"]
harvest = one_layer()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax2.set_title("The Layer L=2")


# vegetables2 = ["Who","made", "these", "wonderful", "cars", "that", "people","drove","as","if", "they", "were","an", "extension", "of", "their", "own","bodies", "?"]
# farmers2 = ["Who","made", "these", "wonderful", "cars", "that", "people","drove","as","if", "they", "were","an", "extension", "of", "their", "own","bodies" "?"]
harvest2 = two_layer()
im = ax2.imshow(harvest2)
ax2.set_xticks(np.arange(len(farmers)))
ax2.set_yticks(np.arange(len(vegetables)))
ax2.set_xticklabels(farmers)
ax2.set_yticklabels(vegetables)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")



# vegetables3 =["Who","made", "these", "wonderful", "cars", "that", "people","drove","as","if", "they", "were","an", "extension", "of", "their", "own","bodies", "?"]
# farmers3 = ["Who","made", "these", "wonderful", "cars", "that", "people","drove","as","if", "they", "were","an", "extension", "of", "their", "own","bodies", "?"]

harvest3 = three_layer()
im = ax3.imshow(harvest3)
ax3.set_xticks(np.arange(len(farmers)))
ax3.set_yticks(np.arange(len(vegetables)))
ax3.set_xticklabels(farmers)
ax3.set_yticklabels(vegetables)
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")



harvest4 = hidden_216()
im = ax4.imshow(harvest4)

ax4.set_xticks(np.arange(len(farmers)))
ax4.set_yticks(np.arange(len(vegetables)))
ax4.set_xticklabels(farmers)
ax4.set_yticklabels(vegetables)
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

harvest5 = hidden_512()
im = ax5.imshow(harvest5)

ax5.set_xticks(np.arange(len(farmers)))
ax5.set_yticks(np.arange(len(vegetables)))
ax5.set_xticklabels(farmers)
ax5.set_yticklabels(vegetables)
plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

harvest6 = hidden_1024()
im = ax6.imshow(harvest6)
ax6.set_xticks(np.arange(len(farmers)))
ax6.set_yticks(np.arange(len(vegetables)))
ax6.set_xticklabels(farmers)
ax6.set_yticklabels(vegetables)

plt.setp(ax6.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

ax.set_title("The Layer L=1")
ax2.set_title("The Layer L=2")
ax3.set_title("The Layer L=3")

ax4.set_title("Hidden units 216")
ax5.set_title("Hidden units 512")
ax6.set_title("Hidden units 1024")


fig.tight_layout()
plt.show()



