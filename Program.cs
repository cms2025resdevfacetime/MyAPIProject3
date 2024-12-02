using Microsoft.EntityFrameworkCore;
using System.Net.Http;
using System.Text.Json;
using System.Runtime.InteropServices;
using Microsoft.AspNetCore.Mvc;

namespace MyAPIProject3
{
    public class Program
    {
        private static bool IsCodeSnackIDE => !RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static int _selectedPort = 5000;

        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Configure services
            builder.Services.AddDbContext<Models.PrimaryDbContext>(options =>
                options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

            builder.Services.AddControllers();
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseAuthorization();
            app.MapControllers();

            // Start API in background thread
            var apiThread = new Thread(() =>
            {
                try
                {
                    app.Run();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"API Error: {ex.Message}");
                }
            });
            apiThread.Start();

            Thread.Sleep(2000); // Wait for API to start

            // Run console interface
            using (var scope = app.Services.CreateScope())
            {
                var context = scope.ServiceProvider.GetRequiredService<Models.PrimaryDbContext>();
                var controller = new Controllers.ModelDbInitsController(context);
                RunConsoleInterface(controller).GetAwaiter().GetResult();
            }
        }

        private static void DrawBorder(string title = "")
        {
            int width = Console.WindowWidth - 2;
            string horizontal = new string('?', width);
            string titleBar = String.Empty;

            Console.ForegroundColor = ConsoleColor.Cyan;
            if (!string.IsNullOrEmpty(title))
            {
                int padding = (width - title.Length) / 2;
                titleBar = '?' + new string(' ', padding - 1) + title + new string(' ', width - padding - title.Length) + '?';
            }

            Console.WriteLine("?" + horizontal + "?");
            if (!string.IsNullOrEmpty(titleBar))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(titleBar);
                Console.ForegroundColor = ConsoleColor.Cyan;
            }
            Console.ResetColor();
        }

        private static void DrawBottomBorder()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("?" + new string('?', Console.WindowWidth - 2) + "?");
            Console.ResetColor();
        }

        private static void DisplayStatusMessage(string message, bool isError = false)
        {
            Console.ForegroundColor = isError ? ConsoleColor.Red : ConsoleColor.Green;
            Console.WriteLine($"\n? STATUS: {message}");
            Console.ResetColor();
        }

        private static void DrawMenuItem(string number, string text, bool isLast = false)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("? ");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write(number);
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write(isLast ? " ??? " : " ??? ");
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine(text);
            Console.ResetColor();
        }

        private static void DisplayProductInfo(object product)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("? ?" + new string('?', 50));

            var json = JsonSerializer.Serialize(product, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            foreach (var line in json.Split('\n'))
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("? ? ");
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine(line);
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("? ?" + new string('?', 50));
            Console.ResetColor();
        }

        private static async Task GetAllModelDbInits(Controllers.ModelDbInitsController controller)
        {
            DisplayStatusMessage("Fetching ModelDbInits...");
            var result = await controller.GetModelDbInits();
            var models = result.Value;

            if (models != null && models.Any())
            {
                DrawBorder("ModelDbInits List");
                foreach (var model in models)
                {
                    DisplayProductInfo(model);
                }
                DisplayStatusMessage($"Found {models.Count()} models");
            }
            else
            {
                DisplayStatusMessage("No models found", true);
            }
        }

        private static async Task GetAllProducts(Controllers.ModelDbInitsController controller)
        {
            DisplayStatusMessage("Fetching products...");
            var result = await controller.GetAllProducts();

            if (result.Result is OkObjectResult okResult && okResult.Value is IEnumerable<object> products)
            {
                if (products.Any())
                {
                    DrawBorder("Products List");
                    foreach (var product in products)
                    {
                        DisplayProductInfo(product);
                    }
                    DisplayStatusMessage($"Found {products.Count()} products");
                }
                else
                {
                    DisplayStatusMessage("No products found", true);
                }
            }
            else
            {
                DisplayStatusMessage("Error retrieving products", true);
            }
        }

        private static async Task RunMachineLearningImplementation(Controllers.ModelDbInitsController controller)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("? Enter ID: ");
            Console.ForegroundColor = ConsoleColor.White;
            if (!int.TryParse(Console.ReadLine(), out int id))
            {
                DisplayStatusMessage("Invalid ID format", true);
                return;
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("? Enter name: ");
            Console.ForegroundColor = ConsoleColor.White;
            var name = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(name))
            {
                DisplayStatusMessage("Name cannot be empty", true);
                return;
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("? Enter product type: ");
            Console.ForegroundColor = ConsoleColor.White;
            var productType = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(productType))
            {
                DisplayStatusMessage("Product type cannot be empty", true);
                return;
            }

            DisplayStatusMessage("Running Machine Learning Implementation...");

            try
            {
                var result = await controller.Machine_Learning_Implementation_One(id, name, productType);

                DrawBorder("Machine Learning Results");
                if (result.Result is OkObjectResult okResult)
                {
                    DisplayProductInfo(okResult.Value);
                }
                DisplayStatusMessage("Machine Learning process completed successfully");
            }
            catch (Exception ex)
            {
                DisplayStatusMessage($"Error: {ex.Message}", true);
                Console.ForegroundColor = ConsoleColor.DarkRed;
                Console.WriteLine($"? Stack Trace: {ex.StackTrace}");
                Console.ResetColor();
            }
        }

        private static async Task RunConsoleInterface(Controllers.ModelDbInitsController controller)
        {
            bool firstRun = true;

            while (true)
            {
                Console.Clear();
                DrawBorder("Machine Learning API Console Interface");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("? Environment: ");
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine(IsCodeSnackIDE ? "CodeSnack IDE" : "Windows");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("? Status: ");
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("Active");

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write("? Time: ");
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine(DateTime.Now.ToString());

                DrawBorder();

                try
                {
                    if (firstRun)
                    {
                        await GetAllProducts(controller);
                        firstRun = false;
                        DrawBottomBorder();
                    }

                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("? Menu Options:");
                    Console.ResetColor();

                    DrawMenuItem("1", "Get All ModelDbInits");
                    DrawMenuItem("2", "Get All Products");
                    DrawMenuItem("3", "Run Machine Learning Implementation");
                    DrawMenuItem("4", "Exit", true);

                    Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write("\n? Enter your choice (1-4): ");
                    Console.ForegroundColor = ConsoleColor.White;

                    var choice = Console.ReadLine();
                    Console.WriteLine();

                    switch (choice)
                    {
                        case "1":
                            await GetAllModelDbInits(controller);
                            break;
                        case "2":
                            await GetAllProducts(controller);
                            break;
                        case "3":
                            await RunMachineLearningImplementation(controller);
                            break;
                        case "4":
                            DisplayStatusMessage("Shutting down...");
                            Thread.Sleep(1000);
                            Environment.Exit(0);
                            return;
                        default:
                            DisplayStatusMessage("Invalid choice. Please try again.", true);
                            break;
                    }
                }
                catch (Exception ex)
                {
                    DisplayStatusMessage($"Error: {ex.Message}", true);
                    Console.ForegroundColor = ConsoleColor.DarkRed;
                    Console.WriteLine($"? Stack Trace: {ex.StackTrace}");
                    Console.ResetColor();
                }

                DrawBottomBorder();
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("Press any key to continue...");
                Console.ResetColor();
                Console.ReadKey();
            }
        }
    }
}