﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Product")]
public partial class Product
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("SubProduct_A")]
    public int? SubProductA { get; set; }

    [Column("SubProduct_B")]
    public int? SubProductB { get; set; }

    [Column("SubProduct_C")]
    public int? SubProductC { get; set; }
}
