import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss


class Stacking(BaseEstimator, TransformerMixin):
	"""
	Scikit-learn compatible API for stacking.
	"""

	def __init__(
		self,
		estimators=None,
		classification=True,
		transform_target_fn=None,
		transform_pred=None,
		stack_type='T1',
		needs_proba=False,
		metric=None,
		n_folds=4,
		stratified=False,
		shuffle=False,
		random_state=0,
		verbose=0
		):
		"""
		Initializer
		:param estimators: list of tuples
			each tuple in the list contains arbitrary unique name and estimator pairs:
			estimators = [('svc', SVC(gamma='auto')),
						   ('rf', RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0))]
		:param classification: boolean
			True - stacking for classification
			False - stacking for regression
		:param transform_target: callable function to transform target variable
		:param transform_pred: callable function to transform prediction
		:param stack_type: str, in ['T1', 'T2']
			'T1' - predict test set in each fold and calculate mean
			'T2' - fit on full train set and prediction test set once
		:param needs_proba: boolean, indicates whether to predict probabilities
		:param metric: callable function, evaluation metric
			if none, then by default:
				sklearn.metrics.mean_squared_error - regression
				sklearn.metrics.accuracy_score - classification with labels
				sklearn.metrics.log_loss - classification with probabilities
		:param n_folds: int, number of folds in cross-validation
		:param stratified: boolean, if true then use stratified folds in cross-validation
		:param shuffle: boolean, indicates whether to perform shuffle before n_fold split
		:param random_state: int, random seed
		:param verbose: int, level of verbosity
		"""
		# check parameters
		# check estimators
		if estimators is None:
			raise ValueError('Estimators is None')
		elif len(estimators) == 0:
			raise ValueError('List of estimators is empty')

		# check classification
		if not isinstance(classification, bool):
			raise ValueError('parameter classification must be bool')

		# check n_folds
		if not isinstance(n_folds, int):
			raise ValueError('parameter n_folds must be int')
		elif n_folds < 2:
			raise ValueError('parameter n_folds must greater than 1')

		# check stack_type
		if not isinstance(stack_type, str):
			raise ValueError('parameter stack_type must be str')
		elif not stack_type in ['T1', 'T2']:
			raise ValueError('parameter stack_type must be either "T1" or "T2"')

		# check verbose
		if not isinstance(verbose, int):
			raise ValueError('parameter verbose must be int')
		elif (verbose < 0) or (verbose > 2):
			raise ValueError('parameter n_folds must be in [0, 1, 2]')

		self.estimators = estimators
		self.classification = classification
		self.transform_target_fn = transform_target_fn
		self.transform_pred = transform_pred
		self.stack_type = stack_type
		self.needs_proba = needs_proba
		self.metric = metric
		self.n_folds = n_folds
		self.stratified = stratified
		self.shuffle = shuffle
		self.random_state = random_state
		self.verbose = verbose

	def fit(self, X, y, sample_weight=None):
		"""
		Fit all first layer estimators
		:param X: 2d numpy array or sparse matrix of shape [n_samples, n_features]
		:param y: 1d numpy array of shape [n_samples]
		:param sample_weight: 1d numpy array of shape [n_sample]
		:return: object, fitted Stacking instance
		"""
		# TODO: check parameters

		# clone estimators
		self.n_estimators = len(self.estimators)
		self.estimators_ = [(name, clone(estimator)) for name, estimator in self.estimators]

		# attributes
		self.train_shape_ = X.shape

		# if classification, set class label numbers
		if self.classification:
			self.n_classes_ = len(np.unique(y))
		else:
			self.n_classes_ = None

		# specify metric
		if self.metric is None and not self.classification:
			self.metric = mean_squared_error
		elif self.metric is None and self.classification:
			# if needs probabilities
			if self.needs_proba:
				self.metric_ = log_loss
			else:
				self.metric_ = accuracy_score
		else:
			self.metric_ = self.metric
		
		# TODO: verbose

		# corss validation folds split
		if self.classification and self.stratified:
			self.kf_ = StratifiedKFold(
				n_splits=self.n_folds, 
				shuffle=self.shuffle, 
				random_state=self.random_state
				)
			self._y_ = y.copy()
		else:
			self.kf_ = KFold(
				n_splits=self.n_folds, 
				shuffle=self.shuffle, 
				random_state=self.random_state
				)
			self._y_ = None

		# output dimension depends on needs_proba
		# if needs_proba is False, output dimension is 1 for classification (and regression)
		# if needs_proba is True, output dimension is [n_classes]
		# indicated by self.prediction_action_
		if self.classification and self.needs_proba:
			self.n_classes_implicit_ = len(np.unique(y))
			self.prediction_action_ = 'predct_proba'
		else:
			self.n_classes_implicit_ = 1
			self.prediction_action_ = 'predict'

		# create empty numpy array for train predictions (OOF)
		s_train = np.zeros(shape=(X.shape[0], self.n_estimators_ * self.n_classes_implicit_))

		# clone estimators for fitting and storing
		self.models_T1_ = []
		self.models_T2_ = None

		for _, estimator in self.estimators_:
			self.models_T1_.append([clone(estimator) for _ in range(self.n_folds)])

		if self.stack_type == 'T2':
			self.models_T2_ = [clone(estimator) for _, estimator in self.estimators_]

		# create empty numpy array to store scores for each estimator and each fold
		self.scores_ np.zeros(shape=(self.n_estimators_, self.n_folds))

		# create empty list of tuple to store (name, mean, std) for each estimator
		self.mean_std_ = []

		# Fitting
		# loop all estimators
		for estimator_counter, (name, estimator) in enumerate(self.estimators_):
			# verbose
			if self.verbose > 0:
				print('estimator {}: [{}: {}]'.format(estimator_counter, name, estimator.__class__.__name__))
			# loop all folds
			for fold_counter, (train_index, test_index) in enumerate(self.kf_.split(X, y)):
				# split training set and test set
				train_X = X[train_index]
				train_y = y[train_index]
				test_X = X[test_index]
				test_y = y[test_index]

				# TODO: split sample weight accordingly

				# fit estimator
				_ = self._estimator_action(
					estimator=self.models_T1_[estimator_counter][fold_counter],
					train_X=train_X,
					train_y=train_y,
					test_X=None,
					sample_weight=None,
					action='fit',
					transfor_fn=self.transform_target_fn
					)

				# predict OOF of train set
				# TODO: predict probabilities
				s_train[test_index, estimator_counter] = self._estimator_action(
					estimator=self.models_T1_[estimator_counter][fold_counter],
					train_X=None,
					train_y=None,
					test_X=test_X,
					action=self.prediction_action_,
					tranform_fn=self.transform_target_fn
					)

				# calculate score
				score = self.metric(test_y, s_train[test_index, estimator_counter])
				self.scores_[estimator_counter, fold_counter] = score

				# TODO: verbose
				if self.verbose > 0:
					print('fold {}: [{}]'.format(fold_counter, score))

			# calculate mean and std
			estimator_name = self.estimators_[estimator_counter][0]
			estimator_mean = np.mean(self.scores_[estimator_counter])
			estimator_std = np.std(self.scores_[estimator_counter])
			self.mean_std_.append((estimator_name, estimator_mean, estimator_std))

			# verbose
			if self.verbose > 0:
				print('mean: [{}], std: [{}]'.format(estimator_mean, estimator_std))

			# fit estimator on full train set
			if self.stack_type == 'T2':
				# verbose
				if self.verbose > 0:
					print('fitting on full train set...')
				_ = self._estimator_action(
					estimator=self.models_T2_[estimator_counter],
					train_X=X,
					train_y=y,
					test_X=None,
					sample_weight=sample_weight,
					action='fit',
					transform=self.transform_target_fn
					)
		return self






