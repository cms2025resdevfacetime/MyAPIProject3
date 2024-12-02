﻿using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Carrier")]
public partial class Carrier
{
    [Key]
    [Column("id")]
    public int Id { get; set; }
}
