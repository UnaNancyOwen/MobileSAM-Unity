using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Sentis;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.Segmentation
{
    /// <summary>
    /// segmentation class for mobile sam
    /// </summary>
    public class SegmentationModel_MobileSAM : IDisposable
    {
        /// <summary>
        /// encoder class for mobile sam
        /// </summary>
        public class Encoder : BaseModel, IDisposable
        {
            public Vector2 resize_ratio = new Vector2(1.0f, 1.0f);

            /// <summary>
            /// create encoder model for mobile sam from sentis file
            /// </summary>
            /// <param name="file_path">model file path</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Encoder(string file_path, BackendType backend_type = BackendType.GPUCompute)
                : base(file_path, backend_type)
            {
                Initialize();
            }

            /// <summary>
            /// create encoder model from stream
            /// </summary>
            /// <param name="stream">mdoel stream</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Encoder(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
                : base(stream, backend_type)
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
            public Tensor<float> Encode(Texture2D image)
            {
                var resize_texture = Resize(image);
                var square_texture = Square(resize_texture);

                resize_ratio = new Vector2(
                    (float)resize_texture.width / (float)image.width,
                    (float)resize_texture.height / (float)image.height
                );

                var image_embeddings = Predict(square_texture).First().Value as Tensor<float>;

                MonoBehaviour.Destroy(resize_texture);
                MonoBehaviour.Destroy(square_texture);

                return image_embeddings;
            }

            /// <summary>
            /// encorde image with split predict over multiple frames
            /// </summary>
            /// <param name="image">input image</param>
            /// <param name="return_action">return callback</param>
            /// <returns>callback function to returns image embeddings</returns>
            public IEnumerator Encode(Texture2D image, Action<Tensor<float>> return_action)
            {
                var resize_texture = Resize(image);
                var square_texture = Square(resize_texture);

                resize_ratio = new Vector2(
                    (float)resize_texture.width / (float)image.width,
                    (float)resize_texture.height / (float)image.height
                );

                var output_tensors = new Dictionary<string, Tensor>();
                yield return CoroutineHandler.StartStaticCoroutine(Predict(square_texture, (outputs) => output_tensors = outputs));
                var image_embeddings = output_tensors.First().Value as Tensor<float>;

                MonoBehaviour.Destroy(resize_texture);
                MonoBehaviour.Destroy(square_texture);

                return_action(image_embeddings);
            }

            private void Initialize()
            {
                SetInputMax(255.0f);
                SetEditedModel(AddPreProcess());
                SetSliceLayers(5); // TODO : automatic adjust number of layers per frame
            }

            private Model AddPreProcess()
            {
                try
                {
                    // INFO : change input data type to Tensor<float> from Tensor<int>, because TextureConverter.ToTensor() can not convert to Tensor<int>.
                    var functional_graph = new FunctionalGraph();
                    var float_tensor = functional_graph.AddInput(DataType.Float, base.runtime_model.inputs[0].shape);
                    var int_tensor = Functional.Int(float_tensor);
                    var predict = Functional.Forward(base.runtime_model, int_tensor)[0];
                    var edited_model = functional_graph.Compile(predict);

                    return edited_model;
                }
                catch (Exception e)
                {
                    throw new Exception($"[error] can not add post process to model for some reason. ({e.Message})");
                }
            }

            private Texture2D Resize(Texture2D image)
            {
                var input_shape = GetInputShapes().First().Value;
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
            private bool is_predicting = false;
            private int layers_per_frame = 1;

            /// <summary>
            /// create decoder model for mobile sam from sentis file
            /// </summary>
            /// <param name="file_path">model file path</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Decoder(string file_path, BackendType backend_type = BackendType.GPUCompute)
                : base(file_path, backend_type)
            {
                Initialize();
            }

            /// <summary>
            /// create decoder model for mobile sam from model asset
            /// </summary>
            /// <param name="model_asset">model asset</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Decoder(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
                : base(model_asset, backend_type)
            {
                Initialize();
            }

            /// <summary>
            /// create decoder model from stream
            /// </summary>
            /// <param name="stream">mdoel stream</param>
            /// <param name="backend_type">backend type for inference engine</param>
            public Decoder(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
                : base(stream, backend_type)
            {
                Initialize();
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
            /// <param name="points">input annotation points</param>
            /// <param name="labels">input annotation labels</param>
            /// <returns>mask tensor</returns>
            public Tensor<float> Decode(Texture2D image, Tensor<float> image_embeddings, List<Vector2> points, List<float> labels)
            {
                var input_tensors = new Dictionary<string, Tensor<float>>();
                worker.SetInput("image_embeddings", image_embeddings);

                var coords = points.SelectMany(point => new float[] { point.x, point.y }).ToArray();
                var point_coords = new Tensor<float>(new TensorShape(1, points.Count, 2), coords);
                worker.SetInput("point_coords", point_coords);

                var point_labels = new Tensor<float>(new TensorShape(1, points.Count), labels.ToArray());
                worker.SetInput("point_labels", point_labels);

                var mask_input = new Tensor<float>(new TensorShape(1, 1, 256, 256), new float[256 * 256]);
                worker.SetInput("mask_input", mask_input);

                var has_mask_input = new Tensor<float>(new TensorShape(1), new float[] { 0.0f });
                worker.SetInput("has_mask_input", has_mask_input);

                var orig_im_size = new Tensor<float>(new TensorShape(2), new float[] { image.height, image.width });
                worker.SetInput("orig_im_size", orig_im_size);

                worker.Schedule();

                var masks = worker.PeekOutput("masks") as Tensor<float>;

                input_tensors?.AllDispose();

                return masks;
            }

            /// <summary>
            /// decorde image embeddings with split predict over multiple frames
            /// </summary>
            /// <param name="image">input image</param>
            /// <param name="image_embeddings">input image embeddings</param>
            /// <param name="points">input annotation points</param>
            /// <param name="labels">input annotation labels</param>
            /// <param name="return_action">return callback</param>
            /// <returns>callback function to returns masks tensor</returns>
            public IEnumerator Decode(Texture2D image, Tensor<float> image_embeddings, List<Vector2> points, List<float> labels, Action<Tensor<float>> return_action)
            {
                var input_tensors = new Dictionary<string, Tensor<float>>();
                worker.SetInput("image_embeddings", image_embeddings);

                var coords = points.SelectMany(point => new float[] { point.x, point.y }).ToArray();
                var point_coords = new Tensor<float>(new TensorShape(1, points.Count, 2), coords);
                worker.SetInput("point_coords", point_coords);

                var point_labels = new Tensor<float>(new TensorShape(1, points.Count), labels.ToArray());
                worker.SetInput("point_labels", point_labels);

                var mask_input = new Tensor<float>(new TensorShape(1, 1, 256, 256), new float[256 * 256]);
                worker.SetInput("mask_input", mask_input);

                var has_mask_input = new Tensor<float>(new TensorShape(1), new float[] { 0.0f });
                worker.SetInput("has_mask_input", has_mask_input);

                var orig_im_size = new Tensor<float>(new TensorShape(2), new float[] { image.height, image.width });
                worker.SetInput("orig_im_size", orig_im_size);

                if (!is_predicting)
                {
                    schedule = worker.ScheduleIterable();
                    is_predicting = true;
                }

                var layers = 0;
                while (schedule.MoveNext())
                {
                    if ((++layers % layers_per_frame) == 0)
                    {
                        yield return null;
                    }
                }

                var masks = worker.PeekOutput("masks") as Tensor<float>;

                input_tensors?.AllDispose();

                is_predicting = false;

                return_action(masks);
            }

            private void Initialize()
            {
                layers_per_frame = runtime_model.layers.Count / 5; // TODO : automatic adjust number of layers per frame
            }
        }

        private Encoder encoder = null;
        private Decoder decoder = null;

        /// <summary>
        /// create segmentation model for mobile sam from sentis file
        /// </summary>
        /// <param name="encoder_path">encoder model file path</param>
        /// <param name="decoder_path">decoder model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel_MobileSAM(string encoder_path, string decoder_path, BackendType backend_type = BackendType.GPUCompute)
        {
            encoder = new Encoder(encoder_path, backend_type);
            decoder = new Decoder(decoder_path, backend_type);
        }

        /// <summary>
        /// create segmentation model for mobile sam from model stream
        /// </summary>
        /// <param name="encoder_stream">encoder model stream</param>
        /// <param name="decoder_stream">decoder model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public SegmentationModel_MobileSAM(System.IO.Stream encoder_stream, System.IO.Stream decoder_stream, BackendType backend_type = BackendType.GPUCompute)
        {
            encoder = new Encoder(encoder_stream, backend_type);
            decoder = new Decoder(decoder_stream, backend_type);
        }

        /// <summary>
        /// create segmentation model for mobile sam from model asset
        /// </summary>
        /// <param name="encorder_asset">encoder model asset</param>
        /// <param name="decorder_asset">decoder model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel_MobileSAM(ModelAsset encoder_asset, ModelAsset decoder_asset,BackendType backend_type = BackendType.GPUCompute)
        {
            encoder = new Encoder(encoder_asset, backend_type);
            decoder = new Decoder(decoder_asset, backend_type);
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
        /// apply float16 quantize
        /// </summary>
        public void ApplyQuantize()
        {
            encoder?.ApplyQuantize();
            decoder?.ApplyQuantize();
        }

        /// <summary>
        /// segment area
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="point">anotation point</param>
        /// <returns>segment area texture with binary indices in color.r (segment area is 1)</returns>
        public Texture2D Segment(Texture2D image, Vector2 point)
        {
            var points = new List<Vector2>() { point };
            var labels = new List<float>() { 1.0f }; // 0 for points of outside area, 1 for points of inside area
            return Segment(image, points, labels );
        }

        /// <summary>
        /// segment area
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="rect">anotation bounding box</param>
        /// <returns>segment area texture with binary indices in color.r (segment area is 1)</returns>
        public Texture2D Segment(Texture2D image, Rect rect)
        {
            var points = new List<Vector2>() { new Vector2(rect.xMin, rect.yMin), new Vector2(rect.xMax, rect.yMax) };
            var labels = new List<float>() { 2.0f, 3.0f }; // 2 and 3 for top-left and bottom-right of bounding box
            return Segment(image, points, labels);
        }

        /// <summary>
        /// segment area
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="points">anotation points</param>
        /// <param name="labels">anotation labels</param>
        /// <returns>segment area texture with binary indices in color.r (segment area is 1)</returns>
        /// <remarks>anotation labels are 0 for points of outside area, 1 for points of inside area, 2 and 3 for top-left and bottom-right of bounding box</remarks>
        public Texture2D Segment(Texture2D image, List<Vector2> points, List<float> labels)
        {
            Assert.IsTrue(points.Count == labels.Count);

            // encorde
            var image_embeddings = encoder.Encode(image);
            var resize_ratio = encoder.resize_ratio;

            // decode
            var resize_points = points.Select(point => point * resize_ratio).ToList();
            var masks = decoder.Decode(image, image_embeddings, resize_points, labels);

            // generate mask texture
            var masks_tensor = masks.ReadbackAndClone() as Tensor<float>;
            var masks_values = masks_tensor.AsReadOnlyNativeArray();
            var masks_texture = ToTexture(masks_values, image.width, image.height);

            //var int_array = masks_tensor.AsReadOnlyNativeArray().Reinterpret<int>().ToArray();
            //var input_tensor = new Tensor<int>(masks_tensor.shape, int_array);

            // release tensors
            image_embeddings?.Dispose();
            masks_tensor?.Dispose();
            masks?.Dispose();

            return masks_texture;
        }

        /// <summary>
        /// segment area with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="point">anotation point</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns segment area texture with binary indices in color.r (segment area is 1)</returns>
        public IEnumerator Segment(Texture2D image, Vector2 point, Action<Texture2D> return_action)
        {
            var points = new List<Vector2>() { point };
            var labels = new List<float>() { 1.0f }; // 0 for points of outside area, 1 for points of inside area
            Texture2D masks_texture = null;
            yield return CoroutineHandler.StartStaticCoroutine(Segment(image, points, labels, (output) => masks_texture = output));
            return_action(masks_texture);
        }

        /// <summary>
        /// segment area with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="rect">anotation bounding box</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns segment area texture with binary indices in color.r (segment area is 1)</returns>
        public IEnumerator Segment(Texture2D image, Rect rect, Action<Texture2D> return_action)
        {
            var points = new List<Vector2>() { new Vector2(rect.xMin, rect.yMin), new Vector2(rect.xMax, rect.yMax) };
            var labels = new List<float>() { 2.0f, 3.0f }; // 2 and 3 for top-left and bottom-right of bounding box
            Texture2D masks_texture = null;
            yield return CoroutineHandler.StartStaticCoroutine(Segment(image, points, labels, (output) => masks_texture = output));
            return_action(masks_texture);
        }

        /// <summary>
        /// segment area with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="points">anotation points</param>
        /// <param name="labels">anotation labels</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns segment area texture with binary indices in color.r (segment area is 1)</returns>
        /// <remarks>anotation labels are 0 for points of outside area, 1 for points of inside area, 2 and 3 for top-left and bottom-right of bounding box</remarks>
        public IEnumerator Segment(Texture2D image, List<Vector2> points, List<float> labels, Action<Texture2D> return_action)
        {
            Assert.IsTrue(points.Count == labels.Count);

            // encorde
            Tensor<float> image_embeddings = null;
            yield return CoroutineHandler.StartStaticCoroutine(encoder.Encode(image, (output) => image_embeddings = output));
            var resize_ratio = encoder.resize_ratio;

            // decode
            var resize_points = points.Select(point => point * resize_ratio).ToList();
            Tensor<float> masks = null;
            yield return CoroutineHandler.StartStaticCoroutine(decoder.Decode(image, image_embeddings, resize_points, labels, (output) => masks = output));

            // generate mask texture
            var masks_tensor = masks.ReadbackAndClone();
            var masks_values = masks_tensor.AsReadOnlyNativeArray();
            var masks_texture = ToTexture(masks_values, image.width, image.height);

            // release tensors
            image_embeddings?.Dispose();
            masks_tensor?.Dispose();
            masks?.Dispose();

            return_action(masks_texture);
        }


        private Texture2D ToTexture(Unity.Collections.NativeArray<float>.ReadOnly values, int width, int height)
        {
            var texture = new Texture2D(width, height, TextureFormat.R8, false);
            var colors = new Color32[width * height];
            Parallel.For(0, height, y =>
            {
                var inv_y = height - 1 - y;
                for (var x = 0; x < width; x++)
                {
                    var index = inv_y * width + x;
                    colors[y * width + x].r = (byte)((values[index] > 0.0f) ? 1 : 0); ;
                }
            });

            texture.SetPixels32(colors);
            texture.Apply();

            return texture;
        }
    }
}

