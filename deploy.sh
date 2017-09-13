#~/bin/bash
#cd "${0%/*}"
rm -rf deployment-2.zip
python -m compileall .
zip -rq deployment-2.zip .
aws s3 cp deployment-2.zip s3://grepcv-cnn-models/deployment-2.zip
aws lambda update-function-code --region us-west-2 --function-name   tensorflow-lambda-regression --s3-bucket grepcv-cnn-models --s3-key deployment-2.zip
rm -rf deployment-2.zip

