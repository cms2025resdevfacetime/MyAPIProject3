﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Model_DB_Mute_P1_Sales")]
public partial class ModelDbMuteP1Sale
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("Model_DB_Init_ID")]
    public int? ModelDbInitId { get; set; }

    [Column("DB_Mute_P1_Customer_ID")]
    public int? DbMuteP1CustomerId { get; set; }

    [Column("Model_DB_Mute_P1_Sales_TimeStamp", TypeName = "smalldatetime")]
    public DateTime? ModelDbMuteP1SalesTimeStamp { get; set; }

    public byte[]? Data { get; set; }
}
