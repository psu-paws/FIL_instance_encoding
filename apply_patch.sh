cp patch/changes.patch sliced_score_matching/
cp patch/evaluate_scores.py sliced_score_matching/
cp patch/nice_cifar_ssm_vr.yml sliced_score_matching/configs/nice/
cd sliced_score_matching
git apply changes.patch
