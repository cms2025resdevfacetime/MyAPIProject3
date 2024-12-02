﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Operations_Stage_1")]
[Index("OperationsId", Name = "unq_Operations_Stage_1_Operations_ID", IsUnique = true)]
public partial class OperationsStage1
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("Order_ID")]
    public int? OrderId { get; set; }

    [Column("CSR_Opartational_ID")]
    public int? CsrOpartationalId { get; set; }

    [Column("Operational_ID")]
    public int? OperationalId { get; set; }

    [Column("Customer_ID")]
    public int? CustomerId { get; set; }

    [Column("Sales_ID")]
    public int? SalesId { get; set; }

    [Column("Operations_ID")]
    public int? OperationsId { get; set; }

    [Column("SubService_A")]
    public int? SubServiceA { get; set; }

    [Column("SubService_B")]
    public int? SubServiceB { get; set; }

    [Column("SubService_C")]
    public int? SubServiceC { get; set; }

    [Column("SubProduct_A")]
    public int? SubProductA { get; set; }

    [Column("SubProduct_B")]
    public int? SubProductB { get; set; }

    [Column("SubProduct_C")]
    public int? SubProductC { get; set; }

    [Unicode(false)]
    public string? Data { get; set; }
}
