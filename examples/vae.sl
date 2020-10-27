# two encoders and a decoder
! encode_m <- encode_m @ vae.
! encode_sd <- encode_sd @ vae.
! decode <- decode @ vae.

# !module(vae).

# plus a loss
! ext_loss <- loss @ vae.

# the structure of the evaluation
encode_mean(X ; encode_m[X] @ encode_mean, X) <- input(X).
encode_sdev(X ; encode_sd[X]) <- input(X).

latent_rep(X ; normal[m, sd]) <- encode_mean(X, m), encode_sdev(X, sd).

decode(X ; decode[Z]) <- latent_rep(X, Z).

loss(X, Y ; ext_loss[X, Y]) <- decode(X, Y).

# notating an input
input(data1).

# training query
loss(data1, Y, 0)?