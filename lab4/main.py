de_dataset = ds.ImageFolderDataset(cfg.data_path,
                                   class_indexing={'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4})

transform_img = CV.RandomCropDecodeResize([cfg.image_width,cfg.image_height], scale=(0.08, 1.0), ratio=(0.75, 1.333))  #改变尺寸
hwc2chw_op = CV.HWC2CHW()
type_cast_op = C.TypeCast(mstype.float32)
de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
de_dataset = de_dataset.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=8)
de_dataset = de_dataset.map(input_columns="image", operations=type_cast_op, num_parallel_workers=8)
de_dataset = de_dataset.shuffle(buffer_size=cfg.data_size)
(de_train,de_test)=de_dataset.split([0.8,0.2])

de_train=de_train.batch(cfg.batch_size, drop_remainder=True)
de_train=de_train.repeat(cfg.epoch_size)
de_test=de_test.batch(cfg.batch_size, drop_remainder=True)
de_test=de_test.repeat(cfg.epoch_size)
print('训练数据集数量：',de_train.get_dataset_size()*cfg.batch_size)
print('测试数据集数量：',de_test.get_dataset_size()*cfg.batch_size)

data_next=de_dataset.create_dict_iterator().get_next()
print('通道数/图像长/宽：', data_next['image'].shape)
print('一张图像的标签样式：', data_next['label'])  # 一共5类，用0-4的数字表达类别。

plt.figure()
plt.imshow(data_next['image'][0,...])
plt.colorbar()
plt.grid(False)
plt.show()