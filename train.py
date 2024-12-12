from data.loader import SpectralData
from data.splitter import Splitter


loader = SpectralData()
data = loader.data
X = loader.X(lidar=True)
y = loader.y()
y_cat = loader.y(cat=True)
cov = loader.cov()

splitter = Splitter(X=X, y=y_cat, K=5)
splits = splitter.sample_splits("KF")
