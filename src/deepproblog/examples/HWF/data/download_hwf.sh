mkdir raw
cd raw
gdown  'https://drive.google.com/uc?id=1G07kw-wK-rqbg_85tuB7FNfA49q8lvoy'
unzip HWF.zip
rm -Rf pretrain-sym_net
mv data/* ..
rm HWF.zip
cd ../
rm -rf raw