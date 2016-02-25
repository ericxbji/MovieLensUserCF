#__author__ = 'jixuebin'

from DataSplit import *
from UserSimilarity import *
from Index import *

if __name__ == "__main__":
	r = dict()
	try:
		ratings = open('ratings.dat')
		for each_line in ratings:
			try:
				(userID, movieID, rate, timestamp) = each_line.split('::')
				if userID not in r:
					r[userID] = dict()
				r[userID][movieID] = int(rate)  # 记录一个用户给一部电影的评分
			except ValueError as err:
				print('ValueError' + str(err))
	except IOError as err:
		print("IOError" + str(err))

	(train, cv, test) = SplitData(r)  # 将数据集分为训练集、交叉验证集和测试集，3:1:1

	# UserCF，用户相似度选择
	#	W = InvertedIndex(train)  # 余弦相似度，倒排表实现
	#	W = ImprovedSimilarity(train)  # 改进的余弦相似度
	W = Dist_Similarity(train, r)  # 基于距离的相似度
	#	W = FusionSimilarity(w, dist)   # 基于距离和余弦相似度的加权相似度
	#	W = FusionSimilarity(w, dist)   # 基于距离和改进余弦相似度的加权相似度

	listN = [20, 40, 60]  # 给用户u推荐的item个数
	listK = [20, 40, 80, 160]  # 与用户u兴趣最相似的K个用户
	max_F = 0  # 最优F值
	optim_K = 0  # 最优K值
	optim_N = 0  # 最优N值
	for N in listN:
		for K in listK:
			recall = Recall(train, cv, W, K, r, N)  # 召回率
			precision = Precision(train, cv, W, K, r, N)  # 准确率
			coverage = Coverage(train, cv, W, K, r, N)  # 覆盖率
			popularity = Popularity(train, cv, W, K, r, N)  # 新颖度
			F = 2 * precision * recall / (precision + recall)  # F值
			if F > max_F:
				max_F = F
				optim_K = K
				optim_N = N
			try:
				with open('analysis.txt', 'a') as anafile:
					print(N, K, file=anafile)
					print(recall * 100, file=anafile)
					print(precision * 100, file=anafile)
					print(coverage * 100, file=anafile)
					print(popularity, file=anafile)
					print(F * 100, file=anafile)
			except IOError as err:
				print('File Error: ' + str(err))

	print('The optimal N and K are: ')
	print(optim_N, optim_K)
	recall = Recall(train, test, W, optim_K, r, optim_N)  # 召回率
	precision = Precision(train, test, W, optim_K, r, optim_N)  # 准确率
	coverage = Coverage(train, test, W, optim_K, r, optim_N)  # 覆盖率
	popularity = Popularity(train, test, W, optim_K, r, optim_N)  # 新颖度
	F = 2 * precision * recall / (precision + recall)  # F值
	print('%s: %.2f%%' % ('Recall', recall * 100))
	print('%s: %.2f%%' % ('Precision', precision * 100))
	print('%s: %.2f%%' % ('Coverage', coverage * 100))
	print('%s: %.2f' % ('Popularity', popularity))
	print('%s: %.2f%%' % ('F_value', F * 100))
