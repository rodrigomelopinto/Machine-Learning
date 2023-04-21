import matplotlib.pyplot as plt

x1 = [2,5,10,12,13]
x2 = [2,5,10,30,100,300,1000]
vc_tree = []
vc_bayes = []
vc_mlp = []

for x in x1:
    tree = 3**x
    bayes = 1 + 3*x + x**2
    mlp = x*x + x*1 + x*x + x*1 + x*x + x*1 + 2*x + 2*1
    vc_tree.append(tree)
    vc_bayes.append(bayes)
    vc_mlp.append(mlp)

plt.plot(x1,vc_tree,label= "tree")
plt.plot(x1,vc_bayes,label= "bayes")
plt.plot(x1,vc_mlp,label= "mlp")
plt.xlabel("data dimensionality")
plt.ylabel("vc_dimension")
plt.legend()
plt.show()