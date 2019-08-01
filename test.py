import numpy
import chainer


class CNN(chainer.Chain):

	def __init__(self, train=True):
		super(CNN, self).__init__(
			conv1=L.Convolution2D(1, 32, 3),
			bn1 = L.BatchNormalization(32),
			conv2=L.Convolution2D(32, 64, 3),
			bn2 = L.BatchNormalization(64),
			conv3=L.Convolution2D(64, 128, 3),
			bn3 = L.BatchNormalization(128),
			conv4=L.Convolution2D(128, 256, 3),
			bn4 = L.BatchNormalization(256),
			l1=L.Linear(256, 500),
			bnl1 = L.BatchNormalization(500), 
			l2=L.Linear(500,100),
			bnl2 = L.BatchNormalization(100), 
			l3=L.Linear(100,2),
			lex = L.Linear(256,2)
			)
		self.train = train

    
	def __call__(self, x):
		h = self.conv1(x)
#		h = self.bn1(h)
		h = self.conv2(h)
#		h = self.bn2(h)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 4)
		h = self.conv3(h)
#		h = self.bn3(h)
		h = self.conv4(h)
#		h = self.bn4(h)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 4)
		he = self.lex(h)
		he = F.relu(he)
		h = self.l1(h)
#		h = self.bnl1(h)
		h = F.relu(h)
		h = F.dropout(h, train=self.train, ratio=0.6)
		h = self.l2(h)
#		h = self.bnl2(h)
		h = F.relu(h)
		h = F.dropout(h, train=self.train, ratio=0.5)        
		return self.l3(h) + he



#ニューラルネットワークセットアップ
model = L.Classifier(CNN())
    
#optimizerセットアップ
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.003))
optimizer.add_hook(chainer.optimizer.Lasso(0.003))