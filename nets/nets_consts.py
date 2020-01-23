# for variational dropout
LNALPHA0 = 0
KL1, KL2, KL3 = 0.63576, 1.8732, 1.48695

# neural network architecture
BATCHNORM = True		# use batch normalization
AFFINE = True			# use affine transformation when using batch normalization
HIDDEN = 1024			# nodes per hidden layer
LAYERS_FULL = 8			# number of layers in fully connected network
LAYERS_EMBEDDING = 4	# number of layers in each stack of embedding network