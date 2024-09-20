CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29504}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 是否启用混合精度训练的开关参数
AMP_FLAG=${AMP_FLAG:-""}

# 如果 AMP_FLAG 设置为 "amp"，则在命令行中添加 --amp 参数
if [ "$AMP_FLAG" == "amp" ]; then
    AMP_OPTION="--amp"
else
    AMP_OPTION=""
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
$(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} $AMP_OPTION ${@:3}
