﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Employee_Sales")]
public partial class EmployeeSale
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("Employee_Sales_ID")]
    public int? EmployeeSalesId { get; set; }
}
