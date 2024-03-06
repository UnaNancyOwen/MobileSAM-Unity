using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.Segmentation
{
    namespace MobileSAM
    {
        /// <summary>
        /// encoder class for mobile sam
        /// </summary>
        public class Encoder : BaseModel, IDisposable
        {
            private Vector2 resize_ratio = new Vector2(1.0f, 1.0f);

            /// <summary>
            /// create encoder model for mobile sam from onnx file
            /// </summary>
            /// <param name="file_path">model file path</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Encoder(string file_path, BackendType backend_type = BackendType.GPUCompute)
                : base(file_path, backend_type)
            {
                Initialize();
            }

            /// <summary>
            /// create encoder model for mobile sam from model asset
            /// </summary>
            /// <param name="model_asset">model asset</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Encoder(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
                : base(model_asset, backend_type)
            {
                Initialize();
            }

            /// <summary>
            /// dispose encoder model
            /// </summary>
            public new void Dispose()
            {
                base.Dispose();
            }

            /// <summary>
            /// encorde image
            /// </summary>
            /// <param name="image">input image</param>
            /// <returns>image embeddings</returns>
            public TensorFloat Encode(Texture2D image)
            {
                var resize_texture = Resize(image);
                var square_texture = Square(resize_texture);

                resize_ratio = new Vector2(
                    (float)resize_texture.width / (float)image.width,
                    (float)resize_texture.height / (float)image.height
                );

                var image_embeddings = Predict(square_texture)["output"] as TensorFloat;

                MonoBehaviour.Destroy(resize_texture);
                MonoBehaviour.Destroy(square_texture);

                return image_embeddings;
            }

            public Vector2 GetResizeRatio()
            {
                return resize_ratio;
            }

            private void Initialize()
            {
                SetInputMax(255.0f);
            }

            private Texture2D Resize(Texture2D image)
            {
                var input_shape = GetInputShapes()["input"];
                var scale = input_shape[2] * (1.0f / Math.Max(image.width, image.height));
                var width = (int)(image.width * scale + 0.5f);
                var height = (int)(image.height * scale + 0.5f);

                var render_texture = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32);

                RenderTexture.active = render_texture;
                Graphics.Blit(image, render_texture);

                var resize_texture = new Texture2D(render_texture.width, render_texture.height, TextureFormat.RGBA32, false);
                resize_texture.ReadPixels(new Rect(0, 0, resize_texture.width, resize_texture.height), 0, 0);
                resize_texture.Apply();

                RenderTexture.active = null;
                RenderTexture.ReleaseTemporary(render_texture);

                return resize_texture;
            }

            private Texture2D Square(Texture2D image)
            {
                var size = Math.Max(image.width, image.height);
                var square_texture = new Texture2D(size, size);
                square_texture.SetPixels(0, size - image.height, image.width, image.height, image.GetPixels());
                square_texture.Apply();

                return square_texture;
            }
        }

        /// <summary>
        /// decoder class for mobile sam
        /// </summary>
        public class Decoder : BaseModel, IDisposable
        {
            /// <summary>
            /// create decoder model for mobile sam from onnx file
            /// </summary>
            /// <param name="file_path">model file path</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Decoder(string file_path, BackendType backend_type = BackendType.GPUCompute)
                : base(file_path, backend_type)
            {
            }

            /// <summary>
            /// create decoder model for mobile sam from model asset
            /// </summary>
            /// <param name="model_asset">model asset</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Decoder(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
                : base(model_asset, backend_type)
            {
            }

            /// <summary>
            /// dispose decoder model
            /// </summary>
            public new void Dispose()
            {
                base.Dispose();
            }

            /// <summary>
            /// decorde image embeddings
            /// </summary>
            /// <param name="image">input image</param>
            /// <param name="image_embeddings">input image embeddings</param>
            /// <param name="point">input annotation point</param>
            /// <returns>mask tensor</returns>
            public TensorFloat Decode(Texture2D image, Tensor image_embeddings, Vector2 point)
            {
                var input_tensors = new Dictionary<string, Tensor>();
                input_tensors.Add("image_embeddings", image_embeddings);

                var point_coords = new TensorFloat(new TensorShape(1, 1, 2), new float[] { point.x, point.y });
                input_tensors.Add("point_coords", point_coords);

                var point_labels = new TensorFloat(new TensorShape(1, 1), new float[] { 0.0f });
                input_tensors.Add("point_labels", point_labels);

                var mask_input = new TensorFloat(new TensorShape(1, 1, 256, 256), new float[256 * 256]);
                input_tensors.Add("mask_input", mask_input);

                var has_mask_input = new TensorFloat(new TensorShape(1), new float[] { 0.0f });
                input_tensors.Add("has_mask_input", has_mask_input);

                var orig_im_size = new TensorFloat(new TensorShape(2), new float[] { image.height, image.width });
                input_tensors.Add("orig_im_size", orig_im_size);

                worker.Execute(input_tensors);

                var masks = worker.PeekOutput("masks") as TensorFloat;

                input_tensors?.AllDispose();

                return masks;
            }
        }
    }

    /// <summary>
    /// segmentation class for mobile sam
    /// </summary>
    public class SegmentationModel_MobileSAM : IDisposable
    {
        protected MobileSAM.Encoder encoder = null;
        protected MobileSAM.Decoder decoder = null;

        /// <summary>
        /// create segmentation model for mobile sam from onnx file
        /// </summary>
        /// <param name="encoder_path">encoder model file path</param>
        /// <param name="decoder_path">decoder model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel_MobileSAM(string encoder_path, string decoder_path, BackendType backend_type = BackendType.GPUCompute)
        {
            encoder = new MobileSAM.Encoder(encoder_path, backend_type);
            decoder = new MobileSAM.Decoder(decoder_path, backend_type);
        }

        /// <summary>
        /// create segmentation model for mobile sam from model asset
        /// </summary>
        /// <param name="encorder_asset">encoder model asset</param>
        /// <param name="decorder_asset">decoder model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel_MobileSAM(ModelAsset encoder_asset, ModelAsset decoder_asset,BackendType backend_type = BackendType.GPUCompute)
        {
            encoder = new MobileSAM.Encoder(encoder_asset, backend_type);
            decoder = new MobileSAM.Decoder(decoder_asset, backend_type);
        }

        /// <summary>
        /// dispose segmentation model
        /// </summary>
        public void Dispose()
        {
            encoder?.Dispose();
            encoder = null;

            decoder?.Dispose();
            decoder = null;
        }

        /// <summary>
        /// segment area
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="point">anotation point</param>
        /// <returns>segment area texture with binary indices in color.r (segment area is 1)</returns>
        public Texture2D Segment(Texture2D image, Vector2 point)
        {
            // encorde
            var image_embeddings = encoder.Encode(image);
            var resize_ratio = encoder.GetResizeRatio();

            // decode
            var resize_point = point * resize_ratio;
            var masks = decoder.Decode(image, image_embeddings, resize_point);

            // generate mask texture
            masks.MakeReadable();
            var masks_values = masks.ToReadOnlyArray();
            var masks_texture = ToTexture(masks_values, image.width, image.height);

            // release tensors
            image_embeddings?.Dispose();
            masks?.Dispose();

            return masks_texture;
        }

        private Texture2D ToTexture(float[] tensor, int width, int height)
        {
            var texture = new Texture2D(width, height, TextureFormat.R8, false);
            var colors = new Color32[width * height];
            Parallel.For(0, height, y =>
            {
                var inv_y = height - 1 - y;
                for (var x = 0; x < width; x++)
                {
                    var index = inv_y * width + x;
                    colors[y * width + x].r = (byte)((tensor[index] > 0.0f) ? 1 : 0); ;
                }
            });

            texture.SetPixels32(colors);
            texture.Apply();

            return texture;
        }
    }
}
