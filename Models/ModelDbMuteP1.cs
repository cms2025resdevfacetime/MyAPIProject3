﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Model_DB_Mute_P1")]
public partial class ModelDbMuteP1
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("Model_DB_Init_ID")]
    public int? ModelDbInitId { get; set; }

    [Column("Model_DB_Mute_P1_TimeStamp", TypeName = "smalldatetime")]
    public DateTime? ModelDbMuteP1TimeStamp { get; set; }

    [Column("Model_DB_Mute_P1_Customer_ID")]
    public int? ModelDbMuteP1CustomerId { get; set; }

    [Column("Model_DB_Mute_P1_Catagorical_ID")]
    public int? ModelDbMuteP1CatagoricalId { get; set; }

    [Column("Model_DB_Mute_P1_Sales_ID")]
    public int? ModelDbMuteP1SalesId { get; set; }

    [Column("Model_DB_Mute_P1_Operations_ID")]
    public int? ModelDbMuteP1OperationsId { get; set; }

    [Column("Model_DB_Mute_P1_QA_ID")]
    public int? ModelDbMuteP1QaId { get; set; }

    public byte[]? Data { get; set; }

    [Column("Customer_ID")]
    public int? CustomerId { get; set; }
}
