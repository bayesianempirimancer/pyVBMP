# Author: Jeff Beck
# DAN can! DAN can! DAN can!
# This is an expert learning model that desribes the world as a collection of objects each 
# evolves according the a 3 level Hierarchal Markov Process.  There is also global state variable S
# that (along with actions A) influences the two two state variables for each object.  I like to
# think of the hierarchy of object states as corresponding to object identity (z) and the top level
# object hidden state (s) at the middle level and object position (b) and shape ()
# (S,A) -> plate(N_objects):[(z,s) -> (position/orientation/size, shape/image)] -> (c_i,x_i)
# 
# Here c_i is the assignment of pixel i or megapixel i to object n so technically there 
# is a missing arrow from z to c_i.  Just for fun I added an additional 'inertia' latent s to each
# object that does not impact observations...
