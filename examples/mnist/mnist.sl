!namespace mnist.

# of course our image from the dataset is, indeed, an image
is_image(image).

# the vae model
mean(I ; encode_mean[I]) <- is_image(image).
sdev(I ; encode_sdev[I]) <- is_image(image).

latent(I ; normal[M, S]) <- mean(I, M), sdev(I, S).

decode(I ; decode[Z]) <- latent(I, Z).

# we'll focus on the reconstruction loss - adding kl-div for future work
loss(I, O ; reconstruction_loss[I, O]).

# a query to see the generative story
loss(O, class, L), decode(image, O)?

# and our observation (implicitly parameterized by the dataset decl.)
!evidence(image, class in dataset) loss(image, O, 0.0), decode(image, O).