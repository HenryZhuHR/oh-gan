
readonly SRC_DIR="./"
readonly SERVER_DIR="$DLTRAIN_SERVER_HOSTNAME:~/project/oh-gan/"


echo "Uploading to server: $DLTRAIN_SERVER_HOSTNAME"

rsync -av -e 'ssh -p 23' \
    --exclude-from=scripts/server-upload.list \
    --link-dest $SRC_DIR \
    $SRC_DIR  $SERVER_DIR