using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using MyAPIProject3.Models;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using Accord.MachineLearning;
using Accord.Math;
using System.Runtime.CompilerServices;
using System.Dynamic;
using System.Reflection;
using Tensorflow.Contexts;
using Microsoft.CodeAnalysis.Elfie.Serialization;
using Accord.Math.Distances;
using System.Security.Cryptography.X509Certificates;

namespace MyAPIProject3.Controllers
{
    public class Jit_Memory_Object
    {
        private static readonly ExpandoObject _dynamicStorage = new ExpandoObject();
        private static readonly dynamic _dynamicObject = _dynamicStorage;
        private static RuntimeMethodHandle _jitMethodHandle;

        public static void AddProperty(string propertyName, object value)
        {
            var dictionary = (IDictionary<string, object>)_dynamicStorage;
            dictionary[propertyName] = value;
        }

        public static object GetProperty(string propertyName)
        {
            var dictionary = (IDictionary<string, object>)_dynamicStorage;
            return dictionary.TryGetValue(propertyName, out var value) ? value : null;
        }

        public static dynamic DynamicObject => _dynamicObject;

        public static void SetJitMethodHandle(RuntimeMethodHandle handle)
        {
            _jitMethodHandle = handle;
        }

        public static RuntimeMethodHandle GetJitMethodHandle()
        {
            return _jitMethodHandle;
        }
    }

    [Route("api/[controller]")]
    [ApiController]
    public class ModelDbInitsController : ControllerBase
    {
        private readonly PrimaryDbContext _context;
        private static readonly ConditionalWeakTable<object, Jit_Memory_Object> jitMemory =
            new ConditionalWeakTable<object, Jit_Memory_Object>();

        public static async Task Main(string[] args)
        {
            Console.Write("Enter ID: ");
            if (!int.TryParse(Console.ReadLine(), out int id))
            {
                Console.WriteLine("Invalid ID format");
                return;
            }

            Console.Write("Enter Name: ");
            string name = Console.ReadLine();

            Console.Write("Enter Product Type: ");
            string productType = Console.ReadLine();

            var controller = new ModelDbInitsController(new PrimaryDbContext());
            await controller.Machine_Learning_Implementation_One(id, name, productType);
        }

        public ModelDbInitsController(PrimaryDbContext context)
        {
            _context = context;
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<ModelDbInit>>> GetModelDbInits()
        {
            return await _context.ModelDbInits.ToListAsync();
        }

        [HttpGet("Machine_Learning_Implementation_One/GetAllProducts")]
        public async Task<ActionResult<IEnumerable<object>>> GetAllProducts()
        {
            try
            {
                Console.WriteLine("Starting GetAllProducts operation");

                // Get products from SubProduct A (SubProductum)
                var subProductsA = await _context.SubProductAs
                    .AsNoTracking()
                    .Select(p => new {
                        Source = "SubProduct_A",
                        Id = p.Id,
                        ProductName = p.ProductName,
                        ProductType = p.ProductType,
                        Quantity = p.Quantity,
                        Price = p.Price
                    })
                    .ToListAsync();
                Console.WriteLine($"Retrieved {subProductsA.Count} products from SubProduct A");

                // Get products from SubProduct B
                var subProductsB = await _context.SubProductBs
                    .AsNoTracking()
                    .Select(p => new {
                        Source = "SubProduct_B",
                        Id = p.Id,
                        ProductName = p.ProductName,
                        ProductType = p.ProductType,
                        Quantity = p.Quantity,
                        Price = p.Price
                    })
                    .ToListAsync();
                Console.WriteLine($"Retrieved {subProductsB.Count} products from SubProduct B");

                // Get products from SubProduct C
                var subProductsC = await _context.SubProductCs
                    .AsNoTracking()
                    .Select(p => new {
                        Source = "SubProduct_C",
                        Id = p.Id,
                        ProductName = p.ProductName,
                        ProductType = p.ProductType,
                        Quantity = p.Quantity,
                        Price = p.Price
                    })
                    .ToListAsync();
                Console.WriteLine($"Retrieved {subProductsC.Count} products from SubProduct C");

                // Combine all products
                var allProducts = subProductsA
                    .Concat(subProductsB)
                    .Concat(subProductsC)
                    .ToList();

                Console.WriteLine($"Total products retrieved: {allProducts.Count}");

                return Ok(allProducts);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in GetAllProducts: {ex.Message}");
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
        }

        [HttpPost("Machine_Learning_Implementation_One/{id}/{name}/{productType}")]
        public async Task<ActionResult<ModelDbInit>> Machine_Learning_Implementation_One(int id, string name, string productType)
        {
            try
            {
                tf.enable_eager_execution();
                Console.WriteLine("TensorFlow eager execution enabled");

                ModelDbInit modelInit = new ModelDbInit();
                Console.WriteLine($"Created new model instance for ProductType: {productType}");

                var jitObject = new Jit_Memory_Object();
                jitMemory.Add(this, jitObject);
                Console.WriteLine("JIT memory initialized");

                Jit_Memory_Object.SetJitMethodHandle(MethodBase.GetCurrentMethod().MethodHandle);
                Console.WriteLine("JIT method handle stored");

                // Store ProductType in JIT memory for use across stages
                Jit_Memory_Object.AddProperty("ProductType", productType);
                Console.WriteLine($"ProductType {productType} stored in JIT memory");

                // Stage 1
                Console.WriteLine($"Starting Stage 1 for ProductType: {productType}");
                await ProcessFactoryOne(modelInit, id, name, productType, jitObject);
                Console.WriteLine("Stage 1 completed");

                // Run stages 2 and 3 in parallel
                Console.WriteLine($"Starting Stages 2 and 3 in parallel for ProductType: {productType}");
                await Task.WhenAll(
                    Task.Run(() => {
                        Console.WriteLine("Starting Stage 2");
                        ProcessFactoryTwo(modelInit, id, name, productType, jitObject);
                        Console.WriteLine("Stage 2 completed");
                    }),
                    Task.Run(() => {
                        Console.WriteLine("Starting Stage 3");
                        ProcessFactoryThree(modelInit, id, name, productType, jitObject);
                        Console.WriteLine("Stage 3 completed");
                    })
                );

                // Stage 4
                Console.WriteLine("Starting Stage 4");
                ProcessFactoryFour(modelInit, id, name, productType, jitObject);
                Console.WriteLine("Stage 4 completed");

                Console.WriteLine("Processing completed without database save");
                return Ok(modelInit);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error occurred: {ex.Message}");
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
            finally
            {
                jitMemory.Remove(this);
                Console.WriteLine("JIT memory cleaned up");
            }
        }

        private async Task ProcessFactoryOne(ModelDbInit model, int id, string name, string productType, Jit_Memory_Object jitObject)
        {
            Console.WriteLine($"ProcessFactoryOne: Setting ModelDbInitCatagoricalName to {name}");
            model.ModelDbInitCatagoricalName = name;

            Console.WriteLine("ProcessFactoryOne: Setting ModelDbInitModelData to true");
            model.ModelDbInitModelData = true;

            Console.WriteLine("ProcessFactoryOne: Adding Stage1Complete property");
            Jit_Memory_Object.AddProperty("Stage1Complete", true);

            Console.WriteLine("Fetching Model from database");
            var ML_Model = await _context.ModelDbInits
                .AsNoTracking()
                .FirstOrDefaultAsync(m => m.CustomerId == 2);
            ///TODO !!!!!!: Dyanmically Set Cutomer ID 
            Console.WriteLine($"Model fetch completed: {(ML_Model != null ? "Found" : "Not Found")}");

            if (ML_Model == null)
            {
                /// MODEL NOT FOUND By Customer ID
                Console.WriteLine($"No existing model found for ProductType {productType}. Initializing new model creation.");
                model.ModelDbInitModelData = false;
                Jit_Memory_Object.AddProperty("NewModelRequired", true);

                try
                {
                    Console.WriteLine($"Starting subproduct data collection for ProductType {productType}");

                    var allSubProducts = new List<dynamic>();

                    // Sample Data for new Customer for initial model
                    Console.WriteLine("Fetching SubProduct A data");
                    var subproductsA = await _context.SubProductAs
                        .AsNoTracking()
                        .Where(p => p.ProductType == productType)
                        .Select(p => new {
                            p.ProductName,
                            Price = (float)p.Price
                        })
                        .ToListAsync();
                    allSubProducts.AddRange(subproductsA);
                    Console.WriteLine($"Found {subproductsA.Count} SubProduct A records");

                    Console.WriteLine("Fetching SubProduct B data");
                    var subproductsB = await _context.SubProductBs
                        .AsNoTracking()
                        .Where(p => p.ProductType == productType)
                        .Select(p => new {
                            p.ProductName,
                            Price = (float)p.Price
                        })
                        .ToListAsync();
                    allSubProducts.AddRange(subproductsB);
                    Console.WriteLine($"Found {subproductsB.Count} SubProduct B records");

                    Console.WriteLine("Fetching SubProduct C data");
                    var subproductsC = await _context.SubProductCs
                        .AsNoTracking()
                        .Where(p => p.ProductType == productType)
                        .Select(p => new {
                            p.ProductName,
                            Price = (float)p.Price
                        })
                        .ToListAsync();
                    allSubProducts.AddRange(subproductsC);
                    Console.WriteLine($"Found {subproductsC.Count} SubProduct C records");

                    // Store combined list in JIT memory
                    Jit_Memory_Object.AddProperty("AllSubProducts", allSubProducts);
                    Console.WriteLine($"Total subproducts found: {allSubProducts.Count}");

                    model.ModelDbInitModelData = allSubProducts.Any();

                    Console.WriteLine("Subproduct data collection completed");
                    if (allSubProducts == null)
                    {
                        Console.WriteLine("Aqusition of sample Data Failed, no records found");
                    }
                    else
                    {
                        Console.WriteLine($"Products Sample Data found of Type:{productType}");

                        // Names
                        var allSubProductsNames = allSubProducts.Select(p =>
                            p.ProductName
                        ).ToArray();
                        // Prices
                        var allSubProductsPrices = allSubProducts.Select(p =>
                            p.Price
                        ).ToArray();

                        Console.WriteLine($"Training data initialized. Number of samples: {allSubProductsNames.Length}");
                        Console.WriteLine($"Price range: {allSubProductsPrices.Min()} to {allSubProductsPrices.Max()}");

                        Console.WriteLine("Phase One: Initializing the creation of Neural Network...");
                        Console.WriteLine("Initializing tensors for both prices and names");

                        Tensor priceTrainData;
                        Tensor nameTrainData;
                        try
                        {
                            // Prices
                            priceTrainData = tf.convert_to_tensor(allSubProductsPrices, dtype: TF_DataType.TF_FLOAT);
                            priceTrainData = tf.reshape(priceTrainData, new[] { -1, 1 }); // Reshape to 2D

                            // Names
                            var uniqueNames = allSubProductsNames.Distinct().ToList();
                            var nameToIndex = uniqueNames.Select((name, index) => new { name, index })
                                                           .ToDictionary(x => x.name, x => x.index);

                            var nameIndices = allSubProductsNames.Select(name => nameToIndex[name]).ToArray();

                            var oneHotNames = new float[nameIndices.Length, uniqueNames.Count];
                            for (int i = 0; i < nameIndices.Length; i++)
                            {
                                oneHotNames[i, nameIndices[i]] = 1.0f;
                            }
                            nameTrainData = tf.convert_to_tensor(oneHotNames, dtype: TF_DataType.TF_FLOAT);

                            var combinedTrainData = tf.concat(new[] { priceTrainData, nameTrainData }, axis: 1);
                            Console.WriteLine($"Combined tensor shape: {string.Join(", ", combinedTrainData.shape)}");

                            Console.WriteLine("Initializing model variables for combined features");
                            var inputDim = 1 + uniqueNames.Count;
                            var W = tf.Variable(tf.random.normal(new[] { inputDim, 1 }));
                            var b = tf.Variable(tf.zeros(new[] { 1 }));
                            Console.WriteLine($"Modified W shape: {string.Join(", ", W.shape)}, b shape: {string.Join(", ", b.shape)}");

                            Console.WriteLine("Initializing training parameters");
                            int epochs = 100;
                            float learningRate = 1e-2f;

                            Console.WriteLine("Starting training process with combined features");
                            for (int epoch = 0; epoch < epochs; epoch++)
                            {
                                using (var tape = tf.GradientTape())
                                {
                                    var predictions = tf.matmul(combinedTrainData, W) + b;
                                    var loss = tf.reduce_mean(tf.square(predictions - priceTrainData));

                                    var gradients = tape.gradient(loss, new[] { W, b });
                                    W.assign_sub(gradients[0] * learningRate);
                                    b.assign_sub(gradients[1] * learningRate);

                                    if (epoch % 10 == 0)
                                    {
                                        Console.WriteLine($"Training Epoch {epoch}, Loss: {loss.numpy()}");
                                    }
                                }
                            }

                            Console.WriteLine("Starting model serialization process");
                            using (var memoryStream = new MemoryStream())
                            using (var writer = new BinaryWriter(memoryStream))
                            {
                                var wData = W.numpy().ToArray<float>();
                                writer.Write(wData.Length);
                                foreach (var w in wData)
                                {
                                    writer.Write(w);
                                }
                                Console.WriteLine("Model weights serialized successfully");

                                var bData = b.numpy().ToArray<float>();
                                writer.Write(bData.Length);
                                foreach (var bias in bData)
                                {
                                    writer.Write(bias);
                                }
                                Console.WriteLine("Model bias serialized successfully");

                                Jit_Memory_Object.AddProperty("CustomerId", 2);
                                Jit_Memory_Object.AddProperty("Data", memoryStream.ToArray());
                                Console.WriteLine("Model By Customer ID data saved to in-memory object successfully");

                                var storedCutomerID = Jit_Memory_Object.GetProperty("ModelName");
                                var storedModelData = Jit_Memory_Object.GetProperty("Data") as byte[];
                                Console.WriteLine($"Verification - Customer ID: {storedCutomerID}");
                                Console.WriteLine($"Verification - Data Size: {storedModelData?.Length ?? 0} bytes");

                                var customerId = (int)Jit_Memory_Object.GetProperty("CustomerId");

                                var modelToSave = new ModelDbInit
                                {
                                    CustomerId = customerId,
                                    ModelDbInitTimeStamp = DateTime.Now,
                                    ModelDbInitCatagoricalId = null,
                                    ModelDbInitCatagoricalName = null,
                                    ModelDbInitModelData = true,
                                    Data = memoryStream.ToArray(),
                                    ClientInformation = null
                                };

                                try
                                {
                                    Console.WriteLine("Starting to save new model to database");

                                    var existingModel = await _context.ModelDbInits
                                        .AsNoTracking()
                                        .FirstOrDefaultAsync(m => m.CustomerId == customerId);

                                    if (existingModel != null)
                                    {
                                        Console.WriteLine($"Updating existing model for Customer_ID: {customerId}");
                                        modelToSave.Id = existingModel.Id;
                                        _context.Entry(modelToSave).State = EntityState.Modified;
                                    }
                                    else
                                    {
                                        Console.WriteLine($"Creating new model for Customer_ID: {customerId}");
                                        _context.ModelDbInits.Add(modelToSave);
                                    }

                                    await _context.SaveChangesAsync();
                                    Console.WriteLine($"Model saved successfully. id: {modelToSave.Id}, Customer_ID: {modelToSave.CustomerId}");

                                    model.Id = modelToSave.Id;
                                    model.CustomerId = modelToSave.CustomerId;
                                    model.ModelDbInitTimeStamp = modelToSave.ModelDbInitTimeStamp;
                                    model.ModelDbInitCatagoricalId = modelToSave.ModelDbInitCatagoricalId;
                                    model.ModelDbInitCatagoricalName = modelToSave.ModelDbInitCatagoricalName;
                                    model.ModelDbInitModelData = modelToSave.ModelDbInitModelData;
                                    model.Data = modelToSave.Data;

                                    Jit_Memory_Object.AddProperty("SavedModelId", modelToSave.Id);
                                    Jit_Memory_Object.AddProperty("ModelSaveTime", modelToSave.ModelDbInitTimeStamp);
                                    Jit_Memory_Object.AddProperty("NewModelCreated", existingModel == null);

                                    Console.WriteLine("Model metadata updated in memory");
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"Error saving model to database: {ex.Message}");
                                    throw new Exception("Failed to save model to database", ex);
                                }

                                try
                                {
                                    var savedModel = await _context.ModelDbInits
                                        .AsNoTracking()
                                        .FirstOrDefaultAsync(m => m.Id == modelToSave.Id);

                                    if (savedModel != null)
                                    {
                                        Console.WriteLine($"Model verification successful. Found in database with id: {savedModel.Id}");
                                        Console.WriteLine($"Saved model data size: {savedModel.Data?.Length ?? 0} bytes");
                                    }
                                    else
                                    {
                                        Console.WriteLine("Warning: Could not verify saved model in database");
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"Error verifying saved model: {ex.Message}");
                                }

                                Console.WriteLine("Phase two: Initializing Data K Clustering Implementation");
                                Console.WriteLine($"Found {allSubProducts.Count} products with Type '{productType}' for Data Clustering");

                                var distinctPrices = allSubProductsPrices.Distinct().ToList();
                                if (distinctPrices.Count == 1)
                                {
                                    Console.WriteLine("All prices are identical. Skipping K-means clustering.");
                                    float singlePrice = distinctPrices[0];

                                    Jit_Memory_Object.AddProperty("Centroid_1", singlePrice);
                                    Jit_Memory_Object.AddProperty("Centroid_2", singlePrice);
                                    Jit_Memory_Object.AddProperty("Centroid_3", singlePrice);

                                    Jit_Memory_Object.AddProperty("Largest_Centroid_Value", singlePrice);
                                    Jit_Memory_Object.AddProperty("Largest_Centroid_Index", 0);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Index", 0);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Count", allSubProductsPrices.Length);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Value", singlePrice);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Average", singlePrice);

                                    Console.WriteLine($"Single price value: {singlePrice:F4}");
                                }
                                else
                                {
                                    Console.WriteLine("Extracting prices for clustering");
                                    var prices = allSubProductsPrices.Select(p => new double[] { (double)p }).ToArray();

                                    int numClusters = 3;
                                    int numIterations = 100;

                                    Console.WriteLine($"Clustering parameters: clusters={numClusters}, iterations={numIterations}");

                                    var kmeans = new KMeans(numClusters)
                                    {
                                        MaxIterations = numIterations,
                                        Distance = new SquareEuclidean()
                                    };

                                    Console.WriteLine("Starting k-means clustering");
                                    var clusters = kmeans.Learn(prices);

                                    var centroids = clusters.Centroids;

                                    Console.WriteLine("K-means clustering completed");

                                    var assignments = clusters.Decide(prices);

                                    Console.WriteLine("Final clustering results:");
                                    for (int i = 0; i < prices.Length; i++)
                                    {
                                        Console.WriteLine($"Price: {prices[i][0]:F4}, Cluster: {assignments[i]}");
                                    }

                                    Console.WriteLine("Final centroids:");
                                    for (int i = 0; i < numClusters; i++)
                                    {
                                        Console.WriteLine($"Centroid {i}: {centroids[i][0]:F4}");
                                    }

                                    Jit_Memory_Object.AddProperty("Centroid_1", (float)centroids[0][0]);
                                    Jit_Memory_Object.AddProperty("Centroid_2", (float)centroids[1][0]);
                                    Jit_Memory_Object.AddProperty("Centroid_3", (float)centroids[2][0]);

                                    var Centroid_1 = Jit_Memory_Object.GetProperty("Centroid_1");
                                    var Centroid_2 = Jit_Memory_Object.GetProperty("Centroid_2");
                                    var Centroid_3 = Jit_Memory_Object.GetProperty("Centroid_3");

                                    Console.WriteLine($"Verification - Centroid_1: {Centroid_1}");
                                    Console.WriteLine($"Verification - Centroid_2: {Centroid_2}");
                                    Console.WriteLine($"Verification - Centroid_3: {Centroid_3}");

                                    var centroidValues = centroids.Select((c, i) => new { Value = c[0], Index = i }).ToList();
                                    var largestCentroid = centroidValues.OrderByDescending(c => c.Value).First();

                                    var clusterCounts = assignments.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count());

                                    var largestCluster = clusterCounts.OrderByDescending(kvp => kvp.Value).First();

                                    var pointsInLargestCluster = prices
                                        .Select((p, i) => new { Price = p[0], ClusterIndex = assignments[i] })
                                        .Where(p => p.ClusterIndex == largestCluster.Key)
                                        .Select(p => p.Price)
                                        .ToList();

                                    var largestClusterAverage = pointsInLargestCluster.Average();

                                    Jit_Memory_Object.AddProperty("Largest_Centroid_Value", (float)largestCentroid.Value);
                                    Jit_Memory_Object.AddProperty("Largest_Centroid_Index", largestCentroid.Index);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Index", largestCluster.Key);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Count", largestCluster.Value);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Value", (float)centroids[largestCluster.Key][0]);
                                    Jit_Memory_Object.AddProperty("Most_Points_Cluster_Average", (float)largestClusterAverage);

                                    Console.WriteLine("\n=== Extended Clustering Analysis ===");
                                    Console.WriteLine($"Largest Centroid Value: {Jit_Memory_Object.GetProperty("Largest_Centroid_Value"):F4}");
                                    Console.WriteLine($"Largest Centroid Index: {Jit_Memory_Object.GetProperty("Largest_Centroid_Index")}");
                                    Console.WriteLine($"Cluster with Most Points - Index: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Index")}");
                                    Console.WriteLine($"Cluster with Most Points - Count: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Count")}");
                                    Console.WriteLine($"Cluster with Most Points - Value: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Value"):F4}");
                                    Console.WriteLine($"Cluster with Most Points - Average: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Average"):F4}");
                                    Console.WriteLine("================================\n");

                                    double totalDistance = 0;
                                    for (int i = 0; i < prices.Length; i++)
                                    {
                                        double distance = Math.Abs(prices[i][0] - centroids[assignments[i]][0]);
                                        totalDistance += distance;
                                    }
                                    double avgDistance = totalDistance / prices.Length;

                                    Console.WriteLine($"Average distance to assigned centroid: {avgDistance:F4}");
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Tensor initialization failed: {ex.Message}");
                            throw new Exception("Failed to initialize tensor from price data.", ex);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during subproduct data collection for ProductType {productType}: {ex.Message}");
                    throw;
                }
            }
            else
            {
                Console.WriteLine($"Existing ML Model found for Customer ID {ML_Model.CustomerId}");
                model.ModelDbInitModelData = true;
                Jit_Memory_Object.AddProperty("ExistingModelFound", true);

                // Store model properties in JIT memory
                Jit_Memory_Object.AddProperty("CustomerId", ML_Model.CustomerId);
                Jit_Memory_Object.AddProperty("ModelDbInitTimeStamp", ML_Model.ModelDbInitTimeStamp);
                Jit_Memory_Object.AddProperty("Id", ML_Model.Id);
                Jit_Memory_Object.AddProperty("Data", ML_Model.Data);

                // Verify stored properties
                var storedCustomerId = Jit_Memory_Object.GetProperty("CustomerId");
                var storedTimestamp = Jit_Memory_Object.GetProperty("ModelDbInitTimeStamp");
                var storedId = Jit_Memory_Object.GetProperty("Id");
                var storedData = Jit_Memory_Object.GetProperty("Data");

                Console.WriteLine($"Verification - CustomerId: {storedCustomerId}");
                Console.WriteLine($"Verification - Timestamp: {storedTimestamp}");
                Console.WriteLine($"Verification - Id: {storedId}");
                Console.WriteLine($"Verification - Data Size: {(storedData as byte[])?.Length ?? 0} bytes");

                Console.WriteLine($"Starting subproduct data collection for ProductType {productType}");
                var allSubProducts = new List<dynamic>();

                Console.WriteLine("Fetching SubProduct A data");
                var subproductsA = await _context.SubProductAs
                    .AsNoTracking()
                    .Where(p => p.ProductType == productType)
                    .Select(p => new {
                        p.ProductName,
                        Price = (float)p.Price
                    })
                    .ToListAsync();
                allSubProducts.AddRange(subproductsA);
                Console.WriteLine($"Found {subproductsA.Count} SubProduct A records");

                Console.WriteLine("Fetching SubProduct B data");
                var subproductsB = await _context.SubProductBs
                    .AsNoTracking()
                    .Where(p => p.ProductType == productType)
                    .Select(p => new {
                        p.ProductName,
                        Price = (float)p.Price
                    })
                    .ToListAsync();
                allSubProducts.AddRange(subproductsB);
                Console.WriteLine($"Found {subproductsB.Count} SubProduct B records");

                Console.WriteLine("Fetching SubProduct C data");
                var subproductsC = await _context.SubProductCs
                    .AsNoTracking()
                    .Where(p => p.ProductType == productType)
                    .Select(p => new {
                        p.ProductName,
                        Price = (float)p.Price
                    })
                    .ToListAsync();
                allSubProducts.AddRange(subproductsC);
                Console.WriteLine($"Found {subproductsC.Count} SubProduct C records");

                Jit_Memory_Object.AddProperty("AllSubProducts", allSubProducts);
                Console.WriteLine($"Total subproducts found: {allSubProducts.Count}");
                model.ModelDbInitModelData = allSubProducts.Any();
                Console.WriteLine("Subproduct data collection completed");

                if (allSubProducts == null)
                {
                    Console.WriteLine("Acquisition of sample Data Failed, no records found");
                }
                else
                {
                    Console.WriteLine($"Products Sample Data found of Type:{productType}");

                    var allSubProductsNames = allSubProducts.Select(p =>
                        p.ProductName
                    ).ToArray();
                    var allSubProductsPrices = allSubProducts.Select(p =>
                        p.Price
                    ).ToArray();

                    Console.WriteLine("Phase two: Initializing Data K Clustering Implementation");
                    Console.WriteLine($"Found {allSubProducts.Count} products with Type '{productType}' for Data Clustering");

                    var distinctPrices = allSubProductsPrices.Distinct().ToList();
                    if (distinctPrices.Count == 1)
                    {
                        Console.WriteLine("All prices are identical. Skipping K-means clustering.");
                        float singlePrice = distinctPrices[0];

                        Jit_Memory_Object.AddProperty("Centroid_1", singlePrice);
                        Jit_Memory_Object.AddProperty("Centroid_2", singlePrice);
                        Jit_Memory_Object.AddProperty("Centroid_3", singlePrice);

                        Jit_Memory_Object.AddProperty("Largest_Centroid_Value", singlePrice);
                        Jit_Memory_Object.AddProperty("Largest_Centroid_Index", 0);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Index", 0);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Count", allSubProductsPrices.Length);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Value", singlePrice);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Average", singlePrice);

                        Console.WriteLine($"Single price value: {singlePrice:F4}");
                    }
                    else
                    {
                        Console.WriteLine("Extracting prices for clustering");
                        var prices = allSubProductsPrices.Select(p => new double[] { (double)p }).ToArray();

                        int numClusters = 3;
                        int numIterations = 100;

                        Console.WriteLine($"Clustering parameters: clusters={numClusters}, iterations={numIterations}");

                        var kmeans = new KMeans(numClusters)
                        {
                            MaxIterations = numIterations,
                            Distance = new SquareEuclidean()
                        };

                        Console.WriteLine("Starting k-means clustering");
                        var clusters = kmeans.Learn(prices);

                        var centroids = clusters.Centroids;

                        Console.WriteLine("K-means clustering completed");

                        var assignments = clusters.Decide(prices);

                        Console.WriteLine("Final clustering results:");
                        for (int i = 0; i < prices.Length; i++)
                        {
                            Console.WriteLine($"Price: {prices[i][0]:F4}, Cluster: {assignments[i]}");
                        }

                        Console.WriteLine("Final centroids:");
                        for (int i = 0; i < numClusters; i++)
                        {
                            Console.WriteLine($"Centroid {i}: {centroids[i][0]:F4}");
                        }

                        Jit_Memory_Object.AddProperty("Centroid_1", (float)centroids[0][0]);
                        Jit_Memory_Object.AddProperty("Centroid_2", (float)centroids[1][0]);
                        Jit_Memory_Object.AddProperty("Centroid_3", (float)centroids[2][0]);

                        var Centroid_1 = Jit_Memory_Object.GetProperty("Centroid_1");
                        var Centroid_2 = Jit_Memory_Object.GetProperty("Centroid_2");
                        var Centroid_3 = Jit_Memory_Object.GetProperty("Centroid_3");

                        Console.WriteLine($"Verification - Centroid_1: {Centroid_1}");
                        Console.WriteLine($"Verification - Centroid_2: {Centroid_2}");
                        Console.WriteLine($"Verification - Centroid_3: {Centroid_3}");

                        var centroidValues = centroids.Select((c, i) => new { Value = c[0], Index = i }).ToList();
                        var largestCentroid = centroidValues.OrderByDescending(c => c.Value).First();

                        var clusterCounts = assignments.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count());

                        var largestCluster = clusterCounts.OrderByDescending(kvp => kvp.Value).First();

                        var pointsInLargestCluster = prices
                            .Select((p, i) => new { Price = p[0], ClusterIndex = assignments[i] })
                            .Where(p => p.ClusterIndex == largestCluster.Key)
                            .Select(p => p.Price)
                            .ToList();

                        var largestClusterAverage = pointsInLargestCluster.Average();

                        Jit_Memory_Object.AddProperty("Largest_Centroid_Value", (float)largestCentroid.Value);
                        Jit_Memory_Object.AddProperty("Largest_Centroid_Index", largestCentroid.Index);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Index", largestCluster.Key);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Count", largestCluster.Value);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Value", (float)centroids[largestCluster.Key][0]);
                        Jit_Memory_Object.AddProperty("Most_Points_Cluster_Average", (float)largestClusterAverage);

                        Console.WriteLine("\n=== Extended Clustering Analysis ===");
                        Console.WriteLine($"Largest Centroid Value: {Jit_Memory_Object.GetProperty("Largest_Centroid_Value"):F4}");
                        Console.WriteLine($"Largest Centroid Index: {Jit_Memory_Object.GetProperty("Largest_Centroid_Index")}");
                        Console.WriteLine($"Cluster with Most Points - Index: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Index")}");
                        Console.WriteLine($"Cluster with Most Points - Count: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Count")}");
                        Console.WriteLine($"Cluster with Most Points - Value: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Value"):F4}");
                        Console.WriteLine($"Cluster with Most Points - Average: {Jit_Memory_Object.GetProperty("Most_Points_Cluster_Average"):F4}");
                        Console.WriteLine("================================\n");

                        double totalDistance = 0;
                        for (int i = 0; i < prices.Length; i++)
                        {
                            double distance = Math.Abs(prices[i][0] - centroids[assignments[i]][0]);
                            totalDistance += distance;
                        }
                        double avgDistance = totalDistance / prices.Length;

                        Console.WriteLine($"Average distance to assigned centroid: {avgDistance:F4}");
                    }
                }
            }
        }

        private void ProcessFactoryTwo(ModelDbInit model, int id, string name, string productType, Jit_Memory_Object jitObject)
        {
            Console.WriteLine($"ProcessFactoryTwo: Processing ProductType {productType}");
            model.ModelDbInitModelData = true;

            Console.WriteLine($"ProcessFactoryTwo: Setting CustomerId to {id}");
            model.CustomerId = id;

            var storedId = Jit_Memory_Object.GetProperty("Id");
            var storedCustomerId = Jit_Memory_Object.GetProperty("CustomerId");
            var storedData = Jit_Memory_Object.GetProperty("Data");

            Console.WriteLine($"ProcessFactoryTwo: Retrieved stored Id: {storedId}");
            Console.WriteLine($"ProcessFactoryTwo: Retrieved stored CustomerId: {storedCustomerId}");
            Console.WriteLine($"ProcessFactoryTwo: Retrieved stored Data size: {(storedData as byte[])?.Length ?? 0} bytes");

            var centroid1 = Jit_Memory_Object.GetProperty("Centroid_1");
            var centroid2 = Jit_Memory_Object.GetProperty("Centroid_2");
            var centroid3 = Jit_Memory_Object.GetProperty("Centroid_3");

            Console.WriteLine($"ProcessFactoryTwo: Retrieved Centroid_1: {centroid1}");
            Console.WriteLine($"ProcessFactoryTwo: Retrieved Centroid_2: {centroid2}");
            Console.WriteLine($"ProcessFactoryTwo: Retrieved Centroid_3: {centroid3}");

            var allSubProducts = Jit_Memory_Object.GetProperty("AllSubProducts") as List<dynamic>;
            if (allSubProducts != null)
            {
                Console.WriteLine($"ProcessFactoryTwo: Retrieved AllSubProducts - Count: {allSubProducts.Count}");
                foreach (var product in allSubProducts)
                {
                    Console.WriteLine($"ProcessFactoryTwo: Product - Name: {product.ProductName}, Price: {product.Price}");
                }
            }
            else
            {
                Console.WriteLine("ProcessFactoryTwo: No AllSubProducts found in JIT memory");
            }

            Console.WriteLine("ProcessFactoryTwo: Adding Stage2Complete property");
            Jit_Memory_Object.AddProperty("Stage2Complete", true);
        }

        private void ProcessFactoryThree(ModelDbInit model, int id, string name, string productType, Jit_Memory_Object jitObject)
        {
            Console.WriteLine($"ProcessFactoryThree: Processing ProductType {productType}");
            model.ModelDbInitModelData = true;

            Console.WriteLine($"ProcessFactoryThree: Setting CustomerId to {id}");
            model.CustomerId = id;

            var storedId = Jit_Memory_Object.GetProperty("Id");
            var storedCustomerId = Jit_Memory_Object.GetProperty("CustomerId");
            var storedData = Jit_Memory_Object.GetProperty("Data");

            Console.WriteLine($"ProcessFactoryThree: Retrieved stored Id: {storedId}");
            Console.WriteLine($"ProcessFactoryThree: Retrieved stored CustomerId: {storedCustomerId}");
            Console.WriteLine($"ProcessFactoryThree: Retrieved stored Data size: {(storedData as byte[])?.Length ?? 0} bytes");

            var centroid1 = Jit_Memory_Object.GetProperty("Centroid_1");
            var centroid2 = Jit_Memory_Object.GetProperty("Centroid_2");
            var centroid3 = Jit_Memory_Object.GetProperty("Centroid_3");

            Console.WriteLine($"ProcessFactoryThree: Retrieved Centroid_1: {centroid1}");
            Console.WriteLine($"ProcessFactoryThree: Retrieved Centroid_2: {centroid2}");
            Console.WriteLine($"ProcessFactoryThree: Retrieved Centroid_3: {centroid3}");

            var allSubProducts = Jit_Memory_Object.GetProperty("AllSubProducts") as List<dynamic>;
            if (allSubProducts != null)
            {
                Console.WriteLine($"ProcessFactoryThree: Retrieved AllSubProducts - Count: {allSubProducts.Count}");
                foreach (var product in allSubProducts)
                {
                    Console.WriteLine($"ProcessFactoryThree: Product - Name: {product.ProductName}, Price: {product.Price}");
                }
            }
            else
            {
                Console.WriteLine("ProcessFactoryThree: No AllSubProducts found in JIT memory");
            }

            Console.WriteLine("ProcessFactoryThree: Adding Stage3Complete property");
            Jit_Memory_Object.AddProperty("Stage3Complete", true);
        }

        private void ProcessFactoryFour(ModelDbInit model, int id, string name, string productType, Jit_Memory_Object jitObject)
        {
            Console.WriteLine($"ProcessFactoryFour: Processing ProductType {productType}");
            model.ModelDbInitTimeStamp = DateTime.Now;

            Console.WriteLine("ProcessFactoryFour: Setting ModelDbInitModelData to true");
            model.ModelDbInitModelData = true;

            Console.WriteLine($"ProcessFactoryFour: Setting CustomerId to {id}");
            model.CustomerId = id;

            Console.WriteLine("ProcessFactoryFour: Retrieving completion status of previous stages");
            var stage1Complete = Jit_Memory_Object.GetProperty("Stage1Complete");
            var stage2Complete = Jit_Memory_Object.GetProperty("Stage2Complete");
            var stage3Complete = Jit_Memory_Object.GetProperty("Stage3Complete");

            Console.WriteLine($"ProcessFactoryFour: Stage completion status for ProductType {productType} - Stage1: {stage1Complete}, Stage2: {stage2Complete}, Stage3: {stage3Complete}");
        }
    }
}