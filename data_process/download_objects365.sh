mkdir Objects365-2020/
mkdir Objects365-2020/license/
mkdir Objects365-2020/train/
mkdir -p Objects365-2020/val/images/v1/
mkdir -p Objects365-2020/val/images/v2/
mkdir -p Objects365-2020/test/images/v1/
mkdir -p Objects365-2020/test/images/v2/
mkdir -p Objects365-2020/train/images/v1/
mkdir -p Objects365-2020/train/images/v2/

cd Objects365-2020/

# wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/license/license.txt.tar.gz -P license/
# # wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.json -P train/
wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz -P train/
# wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json -P val/
# wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/sample_2020.json.tar.gz -P val/

# Doesn't work (403 Forbidden)
# wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/test/zhiyuan_objv2_test.tar.gz -P test/

# train
for i in {0..15}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/patch${i}.tar.gz -P train/images/v1
done
for i in {16..50}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/patch${i}.tar.gz -P train/images/v2/
done
# # val
# for i in {0..15}
#   do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/patch${i}.tar.gz -P val/images/v1
# done

# for i in {16..50}
#   do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/patch${i}.tar.gz -P val/images/v2/
# done


# # test
# for i in {0..15}
#   do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/test/images/v1/patch${i}.tar.gz -P test/images/v1/
# done

# for i in {16..50}
#   do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/test/images/v2/patch${i}.tar.gz -P test/images/v2/
# done

# # unzip
for i in {0..15}
  do tar -zxvf train/images/v1/patch${i}.tar.gz -C train/images/v1/
done

for i in {16..50}
  do tar -zxvf train/images/v2/patch${i}.tar.gz -C train/images/v2/
done

# for i in {0..15}
#   do tar -zxvf val/images/v1/patch${i}.tar.gz -C val/images/v1/
# done

# for i in {16..50}
#   do tar -zxvf val/images/v2/patch${i}.tar.gz -C val/images/v2/
# done

# for i in {0..15}
#   do tar -zxvf test/images/v1/patch${i}.tar.gz -C test/images/v1/
# done

# for i in {16..50}
#   do tar -zxvf test/images/v2/patch${i}.tar.gz -C test/images/v2/
# done

# # remove tar.gz
# rm -rf train/patch*.tar.gz
# rm -rf val/images/v1/patch*.tar.gz
# rm -rf val/images/v2/patch*.tar.gz
# rm -rf test/images/v1/patch*.tar.gz
# rm -rf test/images/v2/patch*.tar.gz
# mkdir -p train/images/v1/
# mkdir -p train/images/v2/
# for i in {0..15}
#     do mv images/patch${i} train/images/v1/
# done
# for i in {16..50}
#     do mv images/patch${i} train/images/v2/
# done
# mv *.json train/
# mv *.jsonl train/