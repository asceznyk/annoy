# Locality Sensitive Hashing - Random Projection #

A library that implements RPLSH. The idea is that given a vector space and a query point (vector) we try to find the closest points (neighbours) as fast as possible. 

How do we do this with RPLSH? <br /> 
Points that are closest to each other are more likely to share a subspace, more on this point further. We split the vector space with n hyperplanes and create subspaces. Once several subspaces are created, for a given query point we search the from the subspace it belongs to, effectively reducing the number of points (neighbours) to search from.


