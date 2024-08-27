# mc_text
bash torchrun_test.sh ../configs/mctextnet/segformer_urur.py ../../mmseg_exp/mctxt_urur/iter_112000.pth 1 --work-dir ../../mmseg_exp/mctxt_urur
bash torchrun_test.sh ../configs/mctextnet/segformer_gid.py ../../mmseg_exp/mctxt_gid/iter_96000.pth 1 --work-dir ../../mmseg_exp/mctxt_gid
# baseline
bash torchrun_test.sh ../configs/segformer/segformer-b5_fbp.py /home/rsr/gyl/RS_Code/Paper_Graph/Introduction/segformer_ckpts/fbp.pth 1 --work-dir ../../mmseg_exp/segformerb5_fbp
bash torchrun_test.sh ../configs/segformer/segformer_mit-b5_8xb2-160k_gid-512x512.py /home/rsr/gyl/RS_Code/Paper_Graph/Introduction/segformer_ckpts/mc_gid.pth 1 --work-dir ../../mmseg_exp/segformerb5_gid
# samhq
## keepgsd
bash torchrun_test.sh ../configs/samhqnet/segformer_gid.py  /data1/gyl/RS_Code/mmseg_exp/samhq1_gid/iter_120000.pth 1 --work-dir ../../mmseg_exp/samhq1_gid
bash torchrun_test.sh ../configs/samhqnet/segformer_urur.py  /data1/gyl/RS_Code/mmseg_exp/samhq1_urur/iter_160000.pth 1 --work-dir ../../mmseg_exp/samhq1_urur
## onlysam
bash torchrun_test.sh ../configs/samhqnet/segformer_onlysam_gid.py /data1/gyl/RS_Code/mmseg_exp/onlysam_gid/iter_48000.pth 1 --work-dir ../../mmseg_exp/onlysam_gid
bash torchrun_test.sh ../configs/samhqnet/segformer_onlysam_fbp.py /data1/gyl/RS_Code/mmseg_exp/onlysam_gid/iter_80000.pth 1 --work-dir ../../mmseg_exp/onlysam_fbp
# swin-base 
bash torchrun_test.sh ../configs/swin/swin-base-patch4-window7-gid-512x512.py /data1/gyl/RS_Code/mmseg_exp/swinbase_gid/iter_16000.pth 1 --work-dir ../../mmseg_visual/swinbase_gid
bash torchrun_test.sh ../configs/swin/swin-base-patch4-window7-urur-512x512.py /data1/gyl/RS_Code/mmseg_exp/swinbase_urur/iter_32000.pth 1 --work-dir ../../mmseg_visual/swinbase_urur
# unet
bash torchrun_test.sh ../configs/unet/unet_s5-d16_fcn_4xb4-160k_gid.py /data1/gyl/RS_Code/mmseg_exp/unet_gid/iter_152000.pth 1 --work-dir ../../mmseg_visual/unet_gid
bash torchrun_test.sh ../configs/unet/unet_s5-d16_fcn_4xb4-160k_urur.py /data1/gyl/RS_Code/mmseg_exp/unet_urur/iter_128000.pth 1 --work-dir ../../mmseg_visual/unet_urur
# pspnet
# deeplabv3+
# segformer
bash torchrun_test.sh ../configs/segformer/segformer_mit-b5_8xb2-160k_gid-512x512.py  1 --work-dir ../../mmseg_visual/segformer_gid
bash torchrun_test.sh ../configs/segformer/segformer_mit-b5_8xb2-160k_urur-512x512.py  1 --work-dir ../../mmseg_visual/segformer_urur

