import copy

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
import torch
import cv2
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class AttentionRCNN(torch.nn.Module):
    """
    Model wth dissected rcnn to insert attention layers into
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = build_model(cfg)
        self.p6_attention = torch.nn.MultiheadAttention(2048, 4, batch_first=True).to(self.model.device)
        self.conv = torch.nn.Conv2d(1024, 2048, (5, 5), (4, 4)).to(self.model.device)
        self.pool = torch.nn.MaxPool2d((11, 5)).to(self.model.device)

    def forward(self, batched_inputs):
        """
        Broken down forward to insert attention.
        The inputs are first pre-processed by making everything the same shape
        Then passed through the backbone, which returns 6 encodings taken from different parts of the backbone as a dictionary.
        These features are then passed to the region proposals along with the gt_annos to return the proposals and losses
        The proposals, images and features are then passed to the ROI heads to get the seg, box, cls and get the loss.
        :param batched_inputs:
        BATCHED INPUT STRUCTURE list[Dict], len(batched_inputs)=BATCH_SIZE:
          'file_name': path to the image
          'image_id': index of the image assigned during data-loading
          'height': Height of the image px (before rescaling)
          'width': Width of the image px (before rescaling)
          'image': Tensor of the image (C, H, W)
          'instances': Instances object defining the number of instances, gt_bbox, gt_masks and gt_classes
        :return:
        """
        if not self.training:
            return self.inference(batched_inputs)
        BATCH_SIZE = len(batched_inputs)
        SEQ_L = len(batched_inputs[0]["context_frames"])
        # print(batched_inputs)
        instances = [batched_input["instances"].to(self.model.device) for batched_input in batched_inputs]
        images = [{"image": batched_input["image"], "height": batched_input["height"], "width":batched_input["width"]}
                  for batched_input in batched_inputs
                  ]
        _ = [images.extend(batched_input["context_frames"]) for batched_input in batched_inputs]
        image_list = self.model.preprocess_image(images)
        input_images = self.model.preprocess_image(batched_inputs)
        # Extract features from the backbone
        features = self.model.backbone(image_list.tensor)
        features_tensor = self.conv(features["res4"])
        # values = torch.reshape(torch.flatten(features_tensor[BATCH_SIZE:]), (BATCH_SIZE, SEQ_L, -1))
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = self.pool(features_tensor)
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = torch.squeeze(torch.squeeze(features_tensor, -1), -1)
        # print(f"feature tensor shape: {features_tensor.shape}")
        FEATURE_DIM = features_tensor.shape[-1]
        image_features = features_tensor[:BATCH_SIZE]
        context_features = features_tensor[BATCH_SIZE:]
        values = torch.ones((BATCH_SIZE, SEQ_L, FEATURE_DIM)).to(self.model.device)
        context_features = torch.reshape(context_features, (BATCH_SIZE, SEQ_L, FEATURE_DIM))
        # print(f"Image feature shape: {image_features.shape}")
        # print(f"Context feature shape: {context_features.shape}")
        attn_output, attn_output_weights = self.p6_attention(torch.unsqueeze(image_features, 1), context_features, values)
        # print(f"Attention ouput shape: {attn_output.shape}")
        # print(f"Attention weights shape: {attn_output_weights.shape}")
        # print(f"Feature shape: {features['res4'].shape}")
        features = {"res4": 0.5*features["res4"][:BATCH_SIZE] +
                            0.5*torch.sum(
                                torch.permute(torch.squeeze(attn_output_weights, 1)*torch.permute(torch.reshape(
                                    features["res4"][BATCH_SIZE:],
                                    (BATCH_SIZE, SEQ_L, features["res4"].shape[-3], features["res4"].shape[-2], features["res4"].shape[-1])
                                ), (2, 3, 4, 0, 1)), (3, 4, 0, 1, 2)),
                                dim=1)}
        # print(f"Attention Feature shape: {features['res4'].shape}")
        # get region proposals
        proposals, proposal_losses = self.model.proposal_generator(input_images, features, gt_instances=instances)
        # print(f"Proposals (RPN): {proposals}")

        # Pass the modified features through the ROI heads
        results, detector_losses = self.model.roi_heads(input_images, features, proposals, targets=instances)
        # print("iter complete")
        # print(f"Losses {losses}")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        """

        :param batched_inputs:
        :return:
        """
        assert not self.training
        BATCH_SIZE = len(batched_inputs)
        SEQ_L = len(batched_inputs[0]["context_frames"])
        images = [{"image": batched_input["image"], "height": batched_input["height"], "width": batched_input["width"]}
                  for batched_input in batched_inputs
                  ]
        _ = [images.extend(batched_input["context_frames"]) for batched_input in batched_inputs]
        image_list = self.model.preprocess_image(images)
        input_images = self.model.preprocess_image(batched_inputs)
        # Extract features from the backbone
        features = self.model.backbone(image_list.tensor)
        features_tensor = self.conv(features["res4"])
        # values = torch.reshape(torch.flatten(features_tensor[BATCH_SIZE:]), (BATCH_SIZE, SEQ_L, -1))
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = self.pool(features_tensor)
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = torch.squeeze(torch.squeeze(features_tensor, -1), -1)
        # print(f"feature tensor shape: {features_tensor.shape}")
        FEATURE_DIM = features_tensor.shape[-1]
        image_features = features_tensor[:BATCH_SIZE]
        context_features = features_tensor[BATCH_SIZE:]
        values = torch.ones((BATCH_SIZE, SEQ_L, FEATURE_DIM)).to(self.model.device)
        context_features = torch.reshape(context_features, (BATCH_SIZE, SEQ_L, FEATURE_DIM))
        attn_output, attn_output_weights = self.p6_attention(torch.unsqueeze(image_features, 1), context_features,
                                                             values)
        features = {"res4": 0.5 * features["res4"][:BATCH_SIZE] +
                            0.5 * torch.sum(
            torch.permute(torch.squeeze(attn_output_weights, 1) * torch.permute(torch.reshape(
                features["res4"][BATCH_SIZE:],
                (BATCH_SIZE, SEQ_L, features["res4"].shape[-3], features["res4"].shape[-2], features["res4"].shape[-1])
            ), (2, 3, 4, 0, 1)), (3, 4, 0, 1, 2)),
            dim=1)}
        proposals, _ = self.model.proposal_generator(input_images, features, gt_instances=None)
        # print(f"Proposals (RPN): {proposals}")

        # Pass the modified features through the ROI heads
        results, _ = self.model.roi_heads(input_images, features, proposals, targets=None)
        return GeneralizedRCNN._postprocess(results, batched_inputs, input_images.image_sizes)


class ChannelAttentionRCNN(torch.nn.Module):
    """
    Model wth dissected rcnn to insert attention layers into
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = build_model(cfg)
        self.p6_attention = torch.nn.MultiheadAttention(24*48, 4, batch_first=True).to(self.model.device)

    def forward(self, batched_inputs):
        """
        Broken down forward to insert attention.
        The inputs are first pre-processed by making everything the same shape
        Then passed through the backbone, which returns 6 encodings taken from different parts of the backbone as a dictionary.
        These features are then passed to the region proposals along with the gt_annos to return the proposals and losses
        The proposals, images and features are then passed to the ROI heads to get the seg, box, cls and get the loss.
        :param batched_inputs:
        BATCHED INPUT STRUCTURE list[Dict], len(batched_inputs)=BATCH_SIZE:
          'file_name': path to the image
          'image_id': index of the image assigned during data-loading
          'height': Height of the image px (before rescaling)
          'width': Width of the image px (before rescaling)
          'image': Tensor of the image (C, H, W)
          'instances': Instances object defining the number of instances, gt_bbox, gt_masks and gt_classes
        :return:
        """
        if not self.training:
            return self.inference(batched_inputs)
        BATCH_SIZE = len(batched_inputs)
        SEQ_L = len(batched_inputs[0]["context_frames"])
        # print(batched_inputs)
        instances = [batched_input["instances"].to(self.model.device) for batched_input in batched_inputs]
        images = [{"image": batched_input["image"], "height": batched_input["height"], "width":batched_input["width"]}
                  for batched_input in batched_inputs
                  ]
        _ = [images.extend(batched_input["context_frames"]) for batched_input in batched_inputs]
        image_list = self.model.preprocess_image(images)
        input_images = self.model.preprocess_image(batched_inputs)
        # Extract features from the backbone
        features = self.model.backbone(image_list.tensor)
        features_tensor = features["res4"]
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = torch.reshape(features_tensor, (features_tensor.shape[0], features_tensor.shape[1], -1))
        # print(f"feature tensor shape: {features_tensor.shape}")
        FEATURE_DIM = features_tensor.shape[-1]
        channels = features_tensor.shape[1]
        image_features = features_tensor[:BATCH_SIZE]
        context_features = features_tensor[BATCH_SIZE:]
        context_features = torch.reshape(context_features, (BATCH_SIZE, SEQ_L, channels, FEATURE_DIM))
        context_features = torch.reshape(context_features, (BATCH_SIZE, -1, FEATURE_DIM))
        # print(f"Image feature shape: {image_features.shape}")
        # print(f"Context feature shape: {context_features.shape}")
        attn_output, attn_output_weights = self.p6_attention(image_features, context_features, context_features)
        # print(f"Attention ouput shape: {attn_output.shape}")
        # print(f"Attention weights shape: {attn_output_weights.shape}")
        attn_output = torch.reshape(attn_output, features["res4"][:BATCH_SIZE].shape)
        # print(f"Feature shape: {features['res4'].shape}")
        features = {"res4": 0.5*features["res4"][:BATCH_SIZE] + 0.5*attn_output}
        # print(f"Attention Feature shape: {features['res4'].shape}")
        # get region proposals
        proposals, proposal_losses = self.model.proposal_generator(input_images, features, gt_instances=instances)
        # print(f"Proposals (RPN): {proposals}")

        # Pass the modified features through the ROI heads
        results, detector_losses = self.model.roi_heads(input_images, features, proposals, targets=instances)
        # print("iter complete")
        # print(f"Losses {losses}")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        """

        :param batched_inputs:
        :return:
        """
        assert not self.training
        BATCH_SIZE = len(batched_inputs)
        SEQ_L = len(batched_inputs[0]["context_frames"])
        # print(batched_inputs)
        # instances = [batched_input["instances"].to(self.model.device) for batched_input in batched_inputs]
        images = [{"image": batched_input["image"], "height": batched_input["height"], "width": batched_input["width"]}
                  for batched_input in batched_inputs
                  ]
        _ = [images.extend(batched_input["context_frames"]) for batched_input in batched_inputs]
        image_list = self.model.preprocess_image(images)
        input_images = self.model.preprocess_image(batched_inputs)
        # Extract features from the backbone
        features = self.model.backbone(image_list.tensor)
        features_tensor = features["res4"]

        features_tensor = torch.reshape(features_tensor, (features_tensor.shape[0], features_tensor.shape[1], -1))
        FEATURE_DIM = features_tensor.shape[-1]
        channels = features_tensor.shape[1]
        image_features = features_tensor[:BATCH_SIZE]
        context_features = features_tensor[BATCH_SIZE:]
        context_features = torch.reshape(context_features, (BATCH_SIZE, SEQ_L, channels, FEATURE_DIM))
        context_features = torch.reshape(context_features, (BATCH_SIZE, -1, FEATURE_DIM))
        attn_output, attn_output_weights = self.p6_attention(image_features, context_features, context_features)
        attn_output = torch.reshape(attn_output, features["res4"][:BATCH_SIZE].shape)
        features = {"res4": 0.5 * features["res4"][:BATCH_SIZE] + 0.5 * attn_output}

        # get region proposals
        proposals, _ = self.model.proposal_generator(input_images, features, gt_instances=None)
        # print(f"Proposals (RPN): {proposals}")

        # Pass the modified features through the ROI heads
        results, _ = self.model.roi_heads(input_images, features, proposals, targets=None)

        return GeneralizedRCNN._postprocess(results, batched_inputs, input_images.image_sizes)


class ChannelSelfAttentionRCNN(torch.nn.Module):
    """
    Model wth dissected rcnn to insert attention layers into
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = build_model(cfg)
        self.p6_attention = torch.nn.MultiheadAttention(1024, 4, batch_first=True).to(self.model.device)
        self.key = torch.nn.Linear(48 * 24, 1024).to(self.model.device)
        self.query = torch.nn.Linear(48 * 24, 1024).to(self.model.device)
        self.value = torch.nn.Linear(48 * 24, 1024).to(self.model.device)
        self.attn_out = torch.nn.Linear(1024, 48*24).to(self.model.device)

    def forward(self, batched_inputs):
        """
        Broken down forward to insert attention.
        The inputs are first pre-processed by making everything the same shape
        Then passed through the backbone, which returns 6 encodings taken from different parts of the backbone as a dictionary.
        These features are then passed to the region proposals along with the gt_annos to return the proposals and losses
        The proposals, images and features are then passed to the ROI heads to get the seg, box, cls and get the loss.
        :param batched_inputs:
        BATCHED INPUT STRUCTURE list[Dict], len(batched_inputs)=BATCH_SIZE:
          'file_name': path to the image
          'image_id': index of the image assigned during data-loading
          'height': Height of the image px (before rescaling)
          'width': Width of the image px (before rescaling)
          'image': Tensor of the image (C, H, W)
          'instances': Instances object defining the number of instances, gt_bbox, gt_masks and gt_classes
        :return:
        """
        if not self.training:
            return self.inference(batched_inputs)
        BATCH_SIZE = len(batched_inputs)
        SEQ_L = len(batched_inputs[0]["context_frames"])
        # print(batched_inputs)
        instances = [batched_input["instances"].to(self.model.device) for batched_input in batched_inputs]
        images = []
        for batched_input in batched_inputs:
            images.extend(batched_input["context_frames"])
            images.append({"image": batched_input["image"], "height": batched_input["height"], "width": batched_input["width"]})
        image_list = self.model.preprocess_image(images)
        input_images = self.model.preprocess_image(batched_inputs)
        # Extract features from the backbone
        features = self.model.backbone(image_list.tensor)
        features_tensor = features["res4"]
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = torch.reshape(features_tensor, (features_tensor.shape[0], features_tensor.shape[1], -1))
        # print(f"feature tensor shape: {features_tensor.shape}")
        FEATURE_DIM = features_tensor.shape[-1]
        channels = features_tensor.shape[1]
        image_features = features_tensor[SEQ_L::SEQ_L+1]
        context_features = torch.reshape(features_tensor, (BATCH_SIZE, SEQ_L+1, channels, FEATURE_DIM))
        context_features = torch.reshape(context_features, (BATCH_SIZE, -1, FEATURE_DIM))
        # print(f"Image feature shape: {image_features.shape}")
        # print(f"Context feature shape: {context_features.shape}")
        query = self.query(image_features)
        key = self.key(context_features)
        value = self.value(context_features)
        # print(f"Key feature shape: {key.shape}")
        # print(f"Query feature shape: {query.shape}")
        # print(f"Value feature shape: {value.shape}")
        attn_output, attn_output_weights = self.p6_attention(query, key, value)
        attn_output = self.attn_out(attn_output)
        # print(f"Attention ouput shape: {attn_output.shape}")
        # print(f"Attention weights shape: {attn_output_weights.shape}")
        attn_output = torch.reshape(attn_output, features["res4"][:BATCH_SIZE].shape)
        # print(f"Feature shape: {features['res4'].shape}")
        features = {"res4": attn_output}
        # print(f"Attention Feature shape: {features['res4'].shape}")
        # get region proposals
        proposals, proposal_losses = self.model.proposal_generator(input_images, features, gt_instances=instances)
        # print(f"Proposals (RPN): {proposals}")

        # Pass the modified features through the ROI heads
        results, detector_losses = self.model.roi_heads(input_images, features, proposals, targets=instances)
        # print("iter complete")
        # print(f"Losses {losses}")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        """

        :param batched_inputs:
        :return:
        """
        assert not self.training
        BATCH_SIZE = len(batched_inputs)
        SEQ_L = len(batched_inputs[0]["context_frames"])
        # print(batched_inputs)
        # instances = [batched_input["instances"].to(self.model.device) for batched_input in batched_inputs]
        images = []
        for batched_input in batched_inputs:
            images.extend(batched_input["context_frames"])
            images.append(
                {"image": batched_input["image"], "height": batched_input["height"], "width": batched_input["width"]})
        image_list = self.model.preprocess_image(images)
        input_images = self.model.preprocess_image(batched_inputs)
        # Extract features from the backbone
        features = self.model.backbone(image_list.tensor)
        features_tensor = features["res4"]
        # print(f"feature tensor shape: {features_tensor.shape}")
        features_tensor = torch.reshape(features_tensor, (features_tensor.shape[0], features_tensor.shape[1], -1))
        # print(f"feature tensor shape: {features_tensor.shape}")
        FEATURE_DIM = features_tensor.shape[-1]
        channels = features_tensor.shape[1]
        image_features = features_tensor[SEQ_L::SEQ_L+1]
        context_features = torch.reshape(features_tensor, (BATCH_SIZE, SEQ_L + 1, channels, FEATURE_DIM))
        context_features = torch.reshape(context_features, (BATCH_SIZE, -1, FEATURE_DIM))
        # print(f"Image feature shape: {image_features.shape}")
        # print(f"Context feature shape: {context_features.shape}")
        query = self.query(image_features)
        key = self.key(context_features)
        value = self.value(context_features)
        # print(f"Key feature shape: {key.shape}")
        # print(f"Query feature shape: {query.shape}")
        # print(f"Value feature shape: {value.shape}")
        attn_output, attn_output_weights = self.p6_attention(query, key, value)
        attn_output = self.attn_out(attn_output)
        # print(f"Attention ouput shape: {attn_output.shape}")
        # print(f"Attention weights: {attn_output_weights")
        attn_output = torch.reshape(attn_output, features["res4"][:BATCH_SIZE].shape)
        # print(f"Feature shape: {features['res4'].shape}")
        features = {"res4": attn_output}
        # print(f"Attention Feature shape: {features['res4'].shape}")
        # get region proposals
        proposals, _ = self.model.proposal_generator(input_images, features, gt_instances=None)
        # print(f"Proposals (RPN): {proposals}")

        # Pass the modified features through the ROI heads
        results, _ = self.model.roi_heads(input_images, features, proposals, targets=None)

        return GeneralizedRCNN._postprocess(results, batched_inputs, input_images.image_sizes)


class AttentionTrainer(DefaultTrainer):
    """
    Trainer to initialize the  correct model, and load data with extra arguments.
    """
    def build_model(self, cfg):
        """
        Build Attention RCNN model with the config
        :param cfg: config to build model with
        :return: AttentionRCNN object
        """
        return ChannelSelfAttentionRCNN(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Creates a new dataloader with a Mapper to add in the pre computer encodings
        :param cfg: Config to build the trainer from
        :return:
        """
        video_list = [
            "Dataset/Videos/video_1.mp4",
            "Dataset/Videos/high_lighting_5_vial.mp4",
            "Dataset/Videos/medium_lighting_5_vial.mp4",
            "Dataset/Videos/resized_ubc_small_vial_vid.mp4",
            "Dataset/Videos/dimmed_resized_2023-06-29_RE-HTE_test_UoT_003.mp4",
            "Dataset/Videos/dimmed_resized_2023-06-29_RE-HTE-UoT_test_004.mp4",
        ]
        video_dict = cls.generate_video_list(video_list)
        for key, val in video_dict.items():
            print(f"{key}: frames: {len(val)}")
        mapper = CustomDatasetMapper(cfg, video_dict, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Creates a new dataloader with a Mapper to add in the pre computer encodings
        :param dataset_name:
        :param cfg: Config to build the trainer from
        :return:
        """
        video_list = [
            "Dataset/Videos/video_1.mp4",
            "Dataset/Videos/high_lighting_5_vial.mp4",
            "Dataset/Videos/medium_lighting_5_vial.mp4",
            "Dataset/Videos/resized_ubc_small_vial_vid.mp4",
            "Dataset/Videos/dimmed_resized_2023-06-29_RE-HTE_test_UoT_003.mp4",
            "Dataset/Videos/dimmed_resized_2023-06-29_RE-HTE-UoT_test_004.mp4",
        ]
        video_dict = cls.generate_video_list(video_list)
        mapper = CustomDatasetMapper(cfg, video_dict, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @staticmethod
    def generate_video_list(video_names):
        ret_dict = {}
        for video in video_names:
            cap = cv2.VideoCapture(video)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            ret_dict[video] = frames[:]
        return ret_dict


class AttentionPredictor:
    """
    Predictor to initialize the  correct model, and load data with extra arguments.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = ChannelAttentionRCNN(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, context_frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            # if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
            #    original_image = original_image[:, :, ::-1]
            inputs = []
            # for original_image, context_frames in zip(original_images, context_frame_lists): 
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            input_im = {"image": image, "height": height, "width": width, "context_frames":[]}
            for context_frame in context_frames:
                height, width = context_frame.shape[:2]
                image = self.aug.get_transform(context_frame).apply_image(context_frame)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_im["context_frames"].append({"image": image, "height": height, "width": width})
            inputs.append(input_im)
            predictions = self.model(inputs)
            return predictions[0]


class CustomDatasetMapper(DatasetMapper):
    """
    Mapper to add in the context encodings.
    """
    def __init__(self, cfg, video_dict, is_train=True):
        params = DatasetMapper.from_config(cfg, is_train)
        params["augmentations"] = [
            T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN], cfg.INPUT.MAX_SIZE_TRAIN
            ),
            # T.RandomSaturation(0.8, 1.2),
            T.RandomBrightness(0.8, 1.2),
            # T.RandomContrast(0.8, 1.2),
            # T.RandomLighting(0.5),
        ]
        super().__init__(**params)
        self.augs = T.AugmentationList([
            T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN], cfg.INPUT.MAX_SIZE_TRAIN
            ),
            # T.RandomSaturation(0.8, 1.2),
            T.RandomBrightness(0.8, 1.2),
            # T.RandomContrast(0.8, 1.2),
            # T.RandomLighting(0.5),
        ])
        self.video_dict = video_dict

    def __call__(self, dataset_dict):
        """
        Function to augment the data dictionary to also contain the contest encodings.
        :param dataset_dict: Dictionary containing one entry, with filename, height, width and annotation
        :return: augmented dictionary with precomputed encodings
        """
        # Call the parent class to get the default behavior
        dataset_dict = super().__call__(dataset_dict)
        # Modify the inputs to include extra information
        dataset_dict["context_frames"] = []
        for frame_no in dataset_dict["context"]:
            frame = self.video_dict[dataset_dict["video_path"]][frame_no]
            # cv2.imwrite("./frame_read.jpg", frame)
            x, y, w, h = dataset_dict["vial_pos"]
            vial = frame[y:y+h, x:x + w]
            resized_vial = cv2.resize(vial, (384, 768))
            # cv2.imwrite("./tmp_context.jpg", resized_vial)
            # print(dataset_dict["file_name"])
            height, width = resized_vial.shape[:2]
            aug_input = T.AugInput(resized_vial)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            # image = self.augs.get_transform(resized_vial).apply_image(resized_vial)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            dataset_dict["context_frames"].append(inputs)

        return dataset_dict


class BackboneEncoding(torch.nn.Module):
    """
        Model wth dissected rcnn to insert attention layers into
    """

    def __init__(self, cfg):
        super().__init__()
        self.model = build_model(cfg)

    def forward(self, batched_inputs):
        """
        Broken down forward to insert attention.
        The inputs are first pre-processed by making everything the same shape
        Then passed through the backbone, which returns 6 encodings taken from different parts of the backbone as a dictionary.
        :param batched_inputs:
        BATCHED INPUT STRUCTURE list[Dict], len(batched_inputs)=BATCH_SIZE:
          'file_name': path to the image
          'image_id': index of the image assigned during data-loading
          'height': Height of the image px (before rescaling)
          'width': Width of the image px (before rescaling)
          'image': Tensor of the image (C, H, W)
          'instances': Instances object defining the number of instances, gt_bbox, gt_masks and gt_classes
        :return: Feature encodings
        """

        image_list = self.model.preprocess_image(batched_inputs)
        features = self.model.backbone(image_list.tensor)
        return features


class EncodingGenerator:
    """
    Trainer to initialize the  correct model, and load data with extra arguments.
    Default predictor code with some changes
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = BackboneEncoding(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])
            return predictions
