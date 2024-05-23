#!/bin/zsh

pdfcrop.sh SIAM_one_backup_optimality.pdf SIAM_one_backup_optimality_cropped.pdf
pdfcrop.sh SIAM_one_backup_solution.pdf SIAM_one_backup_solution_cropped.pdf
pdfcrop.sh SIAM_one_backup_shape_error_error.pdf SIAM_one_backup_shape_error_error_cropped.pdf
pdfcrop.sh SIAM_one_backup_solution_error.pdf SIAM_one_backup_solution_error_cropped.pdf

cp SIAM_one_backup_*_cropped.pdf ../../../../papier_sympnet/img/com3/exact_f_1/

pdfcrop.sh SIAM_exp_backup_optimality.pdf SIAM_exp_backup_optimality_cropped.pdf
pdfcrop.sh SIAM_exp_backup_solution.pdf SIAM_exp_backup_solution_cropped.pdf
pdfcrop.sh SIAM_exp_backup_shape_error_error.pdf SIAM_exp_backup_shape_error_error_cropped.pdf
pdfcrop.sh SIAM_exp_backup_solution_error.pdf SIAM_exp_backup_solution_error_cropped.pdf

cp SIAM_exp_backup_*_cropped.pdf ../../../../papier_sympnet/img/com3/f_exp/

pdfcrop.sh param_SIAM_constant_backup_best_optimality_superposition.pdf param_SIAM_constant_backup_best_optimality_superposition_cropped.pdf
pdfcrop.sh param_SIAM_constant_backup_solution_mu_0.55.pdf param_SIAM_constant_backup_solution_mu_0.55_cropped.pdf
pdfcrop.sh param_SIAM_constant_backup_solution_mu_1.45.pdf param_SIAM_constant_backup_solution_mu_1.45_cropped.pdf
pdfcrop.sh param_SIAM_constant_backup_solution_error_mu_0.55.pdf param_SIAM_constant_backup_solution_error_mu_0.55_cropped.pdf
pdfcrop.sh param_SIAM_constant_backup_solution_error_mu_1.45.pdf param_SIAM_constant_backup_solution_error_mu_1.45_cropped.pdf

cp param_SIAM_constant_*_cropped.pdf ../../../../papier_sympnet/img/com3/exact_f_1_param/
