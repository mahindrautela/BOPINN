import scipy.optimize
import numpy as np
import tensorflow as tf

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.

    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: logging metrics.
        progbar: progress bar.
    """

    def __init__(self, model, x_train, y_train, m=10, factr=1e7, pgtol=1e-5, 
                 epsilon=1e-8, maxiter=5000, maxls=50):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.m = m
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

    def set_weights(self, flat_weights):
        """
        Set weights to the model.

        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.

        Args:
            x: input data.

        Returns:
            loss and gradients for weights as tf.Tensor.
        """

        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.

        Args:
            weights: flatten weights.

        Returns:
            loss and gradients for weights as ndarray.
        """

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def callback(self, weights):
        """
        Callback that prints the progress to stdout.

        Args:
            weights: flatten weights.
        """
        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, 
                                      x0=initial_weights,
                                      factr=self.factr,
                                      pgtol=self.pgtol,
                                      epsilon=self.epsilon,
                                      m=self.m, 
                                      maxls=self.maxls, 
                                      maxiter=self.maxiter,
                                      callback=self.callback)
    
        
        # scipy.optimize.least_squares(func = self.evaluate, x0 = initial_weights)
        
        # scipy.optimize.minimize(fun = self.evaluate, 
        #                         x0 = initial_weights,  
        #                         method='L-BFGS-B', 
        #                         jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
        #                         callback = self.callback, 
        #                         options = {'disp': None,
        #                                   'maxcor': 200, 
        #                                   'ftol': 1 * np.finfo(float).eps,  #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
        #                                   'gtol': 5e-5, 
        #                                   'maxfun':  50000, 
        #                                   'maxiter': 1,
        #                                   'iprint': 50,   #print update every 50 iterations
        #                                   'maxls': 50})
        
        # scipy.optimize.minimize(fun=self.evaluate,
        #                         x0=initial_weights, 
        #                         jac=True, 
        #                         method='BFGS',
        #                         callback=self.callback,
        #                         options={'maxiter': 15000, 
        #                                   'maxls': 20})
        
        # scipy.optimize.minimize(fun = self.evaluate, 
        #                         x0 = initial_weights, 
        #                         args=(), 
        #                         method='L-BFGS-B',
        #                         jac= True, 
        #                         callback = self.callback, 
        #                         options = {'disp': None,
        #                                   'maxcor': 200, 
        #                                   'ftol': 1 * np.finfo(float).eps,  #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
        #                                   'gtol': 5e-5, 
        #                                   'maxfun':  50000, 
        #                                   'maxiter': 1,
        #                                   'iprint': 50,   #print update every 50 iterations
        #                                   'maxls': 50})
        
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
