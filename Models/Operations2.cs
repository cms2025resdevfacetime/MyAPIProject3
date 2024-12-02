using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace MyAPIProject3.Models;

[Table("Operations_2")]
public partial class Operations2
{
    [Key]
    [Column("id")]
    public int Id { get; set; }

    [Column("Operational_ID")]
    public int? OperationalId { get; set; }
}
