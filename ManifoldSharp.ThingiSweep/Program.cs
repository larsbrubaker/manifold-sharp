// Copyright 2026 Lars Brubaker
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Thingi10K sweep of the fast convex Dilate/Erode and the boolean, run through
// scripts/run-thingi-sweep.sh (its header documents the options).
//
// The parent enumerates the corpus and runs each mesh in a child process of this same
// executable (`--one`), so a hang or a crash costs one row, not the run. The child also
// holds one CancelToken for the whole mesh, so a slow check reports "timeout" for itself
// and the remaining checks still get their say where the budget allows.
//
// Malformed inputs are classified and skipped - open, non-manifold, self-intersecting -
// so every verdict in the report is about the kernel, not the file.
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;
using ManifoldSharp.Tests;

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

string? root = null;
string outPath = "thingi-sweep.csv";
string? oneZip = null;
int maxFaces = 20000, timeoutS = 120, jobs = 1;
HashSet<string>? onlyIds = null;
for (int i = 0; i < args.Length; i++)
{
	string V() => args[++i];
	switch (args[i])
	{
		case "--root": root = V(); break;
		case "--out": outPath = V(); break;
		case "--max-faces": maxFaces = int.Parse(V(), CultureInfo.InvariantCulture); break;
		case "--timeout-s": timeoutS = int.Parse(V(), CultureInfo.InvariantCulture); break;
		case "--jobs": jobs = int.Parse(V(), CultureInfo.InvariantCulture); break;
		case "--ids": onlyIds = V().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(); break;
		case "--one": oneZip = V(); break;
		default: Console.Error.WriteLine($"unknown option {args[i]}"); return 2;
	}
}

if (oneZip != null)
{
	// Run once, then format: inside the Select this ran the whole mesh once per column.
	Dictionary<string, string> one = Sweep.RunOne(oneZip, timeoutS);
	Console.Out.WriteLine("ROW," + string.Join(",", Sweep.Header.Select(h => Sweep.Csv(one.GetValueOrDefault(h, "")))));
	return 0;
}

if (root == null || !Directory.Exists(root))
{
	Console.Error.WriteLine("--root <Thingi10K meshes dir> is required (the folder holding Thingi10K-meshes-{1,2,3})");
	return 2;
}

return Sweep.RunParent(root, outPath, maxFaces, timeoutS, jobs, onlyIds);

/// <summary>The sweep's parent (corpus walk, child processes, CSV) and child (one mesh).</summary>
internal static class Sweep
{
	/// <summary>Relative volume tolerance for fast vs sweep and for inclusion-exclusion.</summary>
	private const double VolTol = 1e-6;

	/// <summary>Slack for "dilate grows, erode shrinks": exact in principle, float in practice.</summary>
	private const double GrowTol = 1e-9;

	/// <summary>Seconds a child may run past its own deadline before it is killed as hung.</summary>
	private const int KillGraceS = 30;

	public static readonly string[] Header =
	{
		"id", "faces", "class", "seconds", "in_tris", "in_genus", "in_vol", "r",
		"dilate", "dilate_vol", "dilate_genus",
		"dilate_sweep", "dsweep_vol", "dsweep_genus", "dsweep_rel",
		"erode", "erode_vol", "erode_genus",
		"erode_sweep", "esweep_vol", "esweep_genus", "esweep_rel",
		"incl_excl", "ie_rel",
		"message",
	};

	private static readonly string[] Checks = { "dilate", "dilate_sweep", "erode", "erode_sweep", "incl_excl" };

	public static int RunParent(string root, string outPath, int maxFaces, int timeoutS, int jobs, HashSet<string>? onlyIds)
	{
		var zips = Directory.EnumerateFiles(root, "*.stl.zip", SearchOption.AllDirectories)
			.Select(p => (Id: Path.GetFileName(p)[..^".stl.zip".Length], Path: p))
			.Where(z => onlyIds == null || onlyIds.Contains(z.Id))
			.ToList();
		Dictionary<string, int> faceCounts = ModelsJsonFaces(root);

		var done = new HashSet<string>();
		bool fresh = !File.Exists(outPath) || new FileInfo(outPath).Length == 0;
		if (!fresh)
		{
			foreach (string line in File.ReadLines(outPath).Skip(1))
			{
				done.Add(line.Split(',')[0]);
			}
		}

		var work = new List<(string Id, string Path, int Faces)>();
		int overLimit = 0;
		foreach (var z in zips)
		{
			int faces = faceCounts.TryGetValue(z.Id, out int f) ? f : StlFaceCount(z.Path);
			if (faces > maxFaces)
			{
				overLimit++;
			}
			else if (!done.Contains(z.Id))
			{
				work.Add((z.Id, z.Path, faces));
			}
		}

		// Smallest first: the cheap bulk of the corpus reports early, and a resumed run
		// picks up where the sizes left off.
		work = work.OrderBy(w => w.Faces).ThenBy(w => w.Id, StringComparer.Ordinal).ToList();
		Console.WriteLine($"{zips.Count} meshes found, {overLimit} over {maxFaces} faces, {done.Count} already in {outPath}, {work.Count} to run (jobs {jobs}, timeout {timeoutS}s)");

		using (var csv = new StreamWriter(outPath, append: !fresh) { AutoFlush = true })
		{
			if (fresh)
			{
				csv.WriteLine(string.Join(",", Header));
			}

			object gate = new object();
			int n = 0;
			var clock = Stopwatch.StartNew();
			Parallel.ForEach(work, new ParallelOptions { MaxDegreeOfParallelism = jobs }, w =>
			{
				string row = RunChild(w.Id, w.Path, w.Faces, timeoutS);
				lock (gate)
				{
					csv.WriteLine(row);
					n++;
					string[] c = row.Split(',');
					Console.WriteLine($"[{n}/{work.Count} {clock.Elapsed:hh\\:mm\\:ss}] {w.Id} ({w.Faces}f) {c[2]} {string.Join(" ", Checks.Select(k => $"{k}={c[Array.IndexOf(Header, k)]}"))} {c[3]}s");
				}
			});
		}

		PrintSummary(outPath);
		return 0;
	}

	private static string RunChild(string id, string zip, int faces, int timeoutS)
	{
		var psi = new ProcessStartInfo(Environment.ProcessPath!)
		{
			RedirectStandardOutput = true,
			RedirectStandardError = true,
			UseShellExecute = false,
		};

		// Run as `dotnet ManifoldSharp.ThingiSweep.dll` the host is dotnet, not the apphost.
		if (Path.GetFileNameWithoutExtension(Environment.ProcessPath!) == "dotnet")
		{
			psi.ArgumentList.Add(typeof(Sweep).Assembly.Location);
		}

		foreach (string a in new[] { "--one", zip, "--timeout-s", timeoutS.ToString(CultureInfo.InvariantCulture) })
		{
			psi.ArgumentList.Add(a);
		}

		var clock = Stopwatch.StartNew();
		using var p = Process.Start(psi)!;
		Task<string> stdout = p.StandardOutput.ReadToEndAsync();
		Task<string> stderr = p.StandardError.ReadToEndAsync();
		string cls;
		string message;
		if (!p.WaitForExit((timeoutS + KillGraceS) * 1000))
		{
			p.Kill(entireProcessTree: true);
			p.WaitForExit();
			cls = "hung";
			message = $"killed after {timeoutS + KillGraceS}s in {LastStage(stderr)}";
		}
		else
		{
			string row = stdout.Result.Split('\n').FirstOrDefault(l => l.StartsWith("ROW,", StringComparison.Ordinal))?.TrimEnd('\r') ?? "";
			if (row.Length > 0)
			{
				// The child reports the STL's face count from the weld; the parent's is the
				// file's own and is what --max-faces filtered on, so it wins.
				string[] cells = SplitCsv(row[4..]);
				cells[1] = faces.ToString(CultureInfo.InvariantCulture);
				return string.Join(",", cells.Select(Csv));
			}

			cls = "crash";
			message = $"exit {p.ExitCode}: " + string.Join(" ", stderr.Result.Split('\n').Where(l => !l.StartsWith("STAGE ", StringComparison.Ordinal)).Take(3)).Trim() + $" (in {LastStage(stderr)})";
		}

		var r = new Dictionary<string, string>
		{
			["id"] = id,
			["faces"] = faces.ToString(CultureInfo.InvariantCulture),
			["class"] = cls,
			["seconds"] = clock.Elapsed.TotalSeconds.ToString("F1", CultureInfo.InvariantCulture),
			["message"] = message,
		};
		return string.Join(",", Header.Select(h => Csv(r.GetValueOrDefault(h, ""))));
	}

	/// <summary>The last "STAGE name" line a killed child wrote, to say which step hung.</summary>
	private static string LastStage(Task<string> stderr)
	{
		// The pipe closes with the killed process, so the read completes promptly.
		string text = stderr.Wait(5000) ? stderr.Result : "";
		return text.Split('\n').LastOrDefault(l => l.StartsWith("STAGE ", StringComparison.Ordinal))?[6..].Trim() ?? "unknown";
	}

	/// <summary>Marks the step a child is entering, on stderr, for <see cref="LastStage"/>.</summary>
	private static void Stage(string name)
	{
		Console.Error.WriteLine("STAGE " + name);
		Console.Error.Flush();
	}

	/// <summary>Classify one mesh and, when it is a clean solid, run every check on it.</summary>
	public static Dictionary<string, string> RunOne(string zip, int timeoutS)
	{
		var clock = Stopwatch.StartNew();
		string id = Path.GetFileName(zip)[..^".stl.zip".Length];
		var row = new Dictionary<string, string> { ["id"] = id };
		var msgs = new List<string>();
		try
		{
			RunChecks(zip, timeoutS, row, msgs);
		}
		catch (Exception ex)
		{
			row["class"] = row.ContainsKey("class") ? row["class"] : "import_error";
			msgs.Add(ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
		}

		row["seconds"] = clock.Elapsed.TotalSeconds.ToString("F1", CultureInfo.InvariantCulture);
		row["message"] = string.Join(" | ", msgs);
		return row;
	}

	private static void RunChecks(string zip, int timeoutS, Dictionary<string, string> row, List<string> msgs)
	{
		using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutS));
		var token = new CancelToken(deadline.Token);
		Stage("import");
		byte[] stl = ReadStl(zip);

		MeshGL welded = StlFixtures.ReadStlLikeDemo(stl);
		row["faces"] = (welded.TriVerts.Count / 3).ToString(CultureInfo.InvariantCulture);
		string? edgeClass = EdgeClass(welded);
		if (welded.TriVerts.Count == 0 || edgeClass != null)
		{
			row["class"] = edgeClass ?? "empty";
			return;
		}

		Manifold strict = Manifold.FromMeshGL(welded);
		if (strict.Status() != Error.NoError || strict.IsEmpty())
		{
			// Every edge is paired, yet the strict import refuses: a pinched vertex fan.
			row["class"] = "not_manifold";
			msgs.Add("strict import " + strict.Status());
			return;
		}

		Manifold a = StlFixtures.ImportStlLikeDemo(stl);
		if (a.Status() != Error.NoError || a.IsEmpty())
		{
			row["class"] = "import_error";
			msgs.Add("robust import " + a.Status());
			return;
		}

		// The kernel's own detector: a BVH self-query over the triangles with an exact
		// narrow phase that passes ordinary edge/vertex adjacency and flags crossings,
		// overlaps and coincident (including duplicated) faces. A self-union check misses
		// the self-intersections whose winding happens to come out right.
		Stage("self_intersection");
		if (a.HasSelfIntersections())
		{
			row["class"] = "self_intersecting";
			return;
		}

		Stage("orientation");
		string? misoriented = MisorientedShells(a);
		if (misoriented != null)
		{
			row["class"] = "misoriented";
			msgs.Add(misoriented);
			return;
		}

		row["class"] = "solid";
		Box box = a.BoundingBox();
		Vec3 size = box.Max - box.Min;
		double diag = Math.Sqrt((size.X * size.X) + (size.Y * size.Y) + (size.Z * size.Z));
		double r = 0.02 * diag;
		double inVol = a.Volume();
		int inGenus = a.Genus();
		row["in_tris"] = I(a.NumTri());
		row["in_genus"] = I(inGenus);
		row["in_vol"] = F(inVol);
		row["r"] = F(r);
		Manifold tool = Manifold.Sphere(r, 12);

		// --- fast Dilate invariants, then against the sweep ---
		Stage("dilate");
		bool dilated = a.TryDilateByConvex(tool, token, null, out Manifold fast);
		string dv = FastVerdict(dilated, fast, "dilate", msgs, row);
		if (dv == "ok")
		{
			double v = fast.Volume();
			if (v < inVol * (1 - GrowTol))
			{
				dv = Fail(msgs, $"dilate volume {F(v)} < solid {F(inVol)}");
			}
			else if (!GenusSane(fast.Genus(), inGenus))
			{
				dv = Fail(msgs, $"dilate genus {fast.Genus()} from {inGenus}");
			}
		}

		row["dilate"] = dv;
		Stage("dilate_sweep");
		row["dilate_sweep"] = CompareSweep(dilated && fast.Status() == Error.NoError, fast, () => a.MinkowskiSum(tool, token, null), "dsweep", "dilate", inVol, msgs, row);

		// --- fast Erode, then against the sweep ---
		Stage("erode");
		bool eroded = a.TryErodeByConvex(tool, token, null, out Manifold fastE);
		string ev = FastVerdict(eroded, fastE, "erode", msgs, row, emptyOk: true);
		if (ev == "ok" && fastE.Volume() > inVol * (1 + GrowTol))
		{
			ev = Fail(msgs, $"erode volume {F(fastE.Volume())} > solid {F(inVol)}");
		}

		row["erode"] = ev;
		Stage("erode_sweep");
		row["erode_sweep"] = CompareSweep(eroded && fastE.Status() == Error.NoError, fastE, () => a.MinkowskiDifference(tool, token, null), "esweep", "erode", inVol, msgs, row);

		// --- the boolean on the solid and a shifted copy: vol(A)+vol(B) = vol(A u B)+vol(A n B) ---
		Stage("incl_excl");
		Manifold b = a.Translate(new Vec3(0.3 * size.X, 0.3 * size.Y, 0.3 * size.Z));
		BooleanEngine engine = BooleanConfig.DefaultEngine();
		Manifold u = a.BooleanWithEngineAndToken(b, OpType.Add, engine, token);
		Manifold x = a.BooleanWithEngineAndToken(b, OpType.Intersect, engine, token);
		if (u.Status() == Error.Cancelled || x.Status() == Error.Cancelled)
		{
			row["incl_excl"] = "timeout";
		}
		else if (u.Status() != Error.NoError || x.Status() != Error.NoError)
		{
			row["incl_excl"] = Fail(msgs, $"union {u.Status()} intersect {x.Status()}");
		}
		else
		{
			double vb = b.Volume();
			double ie = Math.Abs(inVol + vb - u.Volume() - x.Volume()) / (inVol + vb);
			row["ie_rel"] = ie.ToString("E3", CultureInfo.InvariantCulture);
			row["incl_excl"] = ie <= VolTol ? "ok" : Fail(msgs, $"A+B-U-I rel {ie:E3}");
		}
	}

	/// <summary>Status, emptiness and a strict re-import of a fast Dilate/Erode result.</summary>
	private static string FastVerdict(bool applied, Manifold m, string name, List<string> msgs, Dictionary<string, string> row, bool emptyOk = false)
	{
		if (!applied)
		{
			return "declined";
		}

		if (m.Status() == Error.Cancelled)
		{
			return "timeout";
		}

		if (m.Status() != Error.NoError || (m.IsEmpty() && !emptyOk))
		{
			return Fail(msgs, $"{name} status {m.Status()} empty {m.IsEmpty()}");
		}

		row[name + "_vol"] = F(m.Volume());
		row[name + "_genus"] = I(m.Genus());
		if (!m.IsEmpty())
		{
			Manifold re = Manifold.FromMeshGL64(m.GetMeshGL64(-1));
			if (re.Status() != Error.NoError || re.IsEmpty())
			{
				return Fail(msgs, $"{name} output does not re-import: {re.Status()}");
			}
		}

		return "ok";
	}

	/// <summary>Volume within <see cref="VolTol"/> and equal genus, fast against sweep.</summary>
	private static string CompareSweep(bool fastOk, Manifold fast, Func<Manifold> sweep, string col, string name, double inVol, List<string> msgs, Dictionary<string, string> row)
	{
		if (!fastOk)
		{
			return "n/a";
		}

		Manifold s = sweep();
		if (s.Status() == Error.Cancelled)
		{
			return "timeout";
		}

		if (s.Status() != Error.NoError)
		{
			return Fail(msgs, $"{name} sweep status {s.Status()}");
		}

		double vf = fast.Volume(), vs = s.Volume();

		// An erosion can legitimately vanish; the floor keeps two near-empty results from
		// dividing noise by noise.
		double rel = Math.Abs(vf - vs) / Math.Max(Math.Max(vf, vs), inVol * 1e-6);
		row[col + "_vol"] = F(vs);
		row[col + "_genus"] = I(s.Genus());
		row[col + "_rel"] = rel.ToString("E3", CultureInfo.InvariantCulture);
		if (rel > VolTol)
		{
			return Fail(msgs, $"{name} fast {F(vf)} sweep {F(vs)} rel {rel:E3}");
		}

		if (fast.Genus() != s.Genus())
		{
			return Fail(msgs, $"{name} genus fast {fast.Genus()} sweep {s.Genus()}");
		}

		return "ok";
	}

	/// <summary>
	/// Null when every shell is wound for the region it bounds, else a description of the
	/// shells ("signed volume @ winding outside it"). Shells do not cross (the input passed
	/// the self-intersection check), so a shell's own point sees only the shells enclosing
	/// it; their winding sum w is the winding just outside it, and w + sign(volume) just
	/// inside. A solid needs both in {0, 1}: an outer shell (w 0) wound positive, a cavity
	/// (w 1) wound negative. Anything else - a reversed outer shell, a cavity wound as
	/// material, a shell doubled inside another - is a malformed file, not a kernel case.
	/// </summary>
	private static string? MisorientedShells(Manifold a)
	{
		List<Manifold> shells = a.Decompose();
		var tris = shells.Select(s => Soup.ImplToTris(s.AsImpl())).ToList();
		var boxes = shells.Select(s => s.BoundingBox()).ToList();
		var bad = new List<string>();
		for (int i = 0; i < shells.Count; i++)
		{
			double v = SignedVolume(tris[i]);
			Vec3 p = tris[i][0][0];
			int w = 0;
			for (int j = 0; j < shells.Count; j++)
			{
				if (j != i && boxes[j].ContainsPoint(p))
				{
					w += RayShoot.WindingNumber(R3.FromVec3(p), tris[j]);
				}
			}

			int sign = Math.Sign(v);
			if (!((w == 0 && sign == 1) || (w == 1 && sign == -1)))
			{
				bad.Add($"shell {i} volume {v:G4} inside winding {w}");
			}
		}

		return bad.Count == 0 ? null : $"{bad.Count} of {shells.Count} shells misoriented: " + string.Join("; ", bad.Take(4));
	}

	/// <summary>
	/// Signed enclosed volume, positive for outward normals. Manifold.Volume() returns the
	/// absolute value (as C++ does), so it cannot tell an inside-out shell from a solid.
	/// </summary>
	private static double SignedVolume(List<Vec3[]> tris)
	{
		double sum = 0;
		foreach (Vec3[] t in tris)
		{
			Vec3 a = t[0], b = t[1], c = t[2];
			sum += (a.X * ((b.Y * c.Z) - (b.Z * c.Y))) + (a.Y * ((b.Z * c.X) - (b.X * c.Z))) + (a.Z * ((b.X * c.Y) - (b.Y * c.X)));
		}

		return sum / 6;
	}

	/// <summary>
	/// A coarse gate only - dilation can both close handles and make them - so it catches a
	/// genus that has gone wild; the exact check is the comparison with the sweep.
	/// </summary>
	private static bool GenusSane(int outGenus, int inGenus)
	{
		return Math.Abs(outGenus) <= (4 * Math.Max(Math.Abs(inGenus), 1)) + 4;
	}

	/// <summary>
	/// "not_closed" when some directed edge has no opposite to pair with, "not_manifold" when
	/// an edge is shared by more than two faces; null when every edge pairs once.
	/// </summary>
	private static string? EdgeClass(MeshGL mesh)
	{
		// Merge welds by recording MergeFromVert -> MergeToVert pairs, not by rewriting
		// TriVerts, so the weld is applied here as the import applies it.
		var weld = new Dictionary<uint, uint>();
		for (int i = 0; i < mesh.MergeFromVert.Count; i++)
		{
			weld[mesh.MergeFromVert[i]] = mesh.MergeToVert[i];
		}

		uint V(uint v) => weld.TryGetValue(v, out uint w) ? w : v;
		var counts = new Dictionary<(uint, uint), int>();
		List<uint> tv = mesh.TriVerts;
		for (int t = 0; t + 2 < tv.Count; t += 3)
		{
			for (int k = 0; k < 3; k++)
			{
				var e = (V(tv[t + k]), V(tv[t + ((k + 1) % 3)]));
				counts[e] = counts.GetValueOrDefault(e) + 1;
			}
		}

		bool overShared = false;
		foreach (var kv in counts)
		{
			if (counts.GetValueOrDefault((kv.Key.Item2, kv.Key.Item1)) != kv.Value)
			{
				return "not_closed";
			}

			overShared |= kv.Value > 1;
		}

		return overShared ? "not_manifold" : null;
	}

	private static byte[] ReadStl(string zip)
	{
		using ZipArchive z = ZipFile.OpenRead(zip);
		ZipArchiveEntry entry = z.Entries.First(e => e.FullName.EndsWith(".stl", StringComparison.OrdinalIgnoreCase));
		using var ms = new MemoryStream();
		using (Stream s = entry.Open())
		{
			s.CopyTo(ms);
		}

		return ms.ToArray();
	}

	/// <summary>
	/// The dataset's own face counts, when a models.json sits in the meshes dir or in the
	/// Thingi10K checkout's docs/data; empty otherwise (each STL is then read to count).
	/// </summary>
	private static Dictionary<string, int> ModelsJsonFaces(string root)
	{
		var result = new Dictionary<string, int>();
		string? path = new[] { Path.Combine(root, "models.json"), Path.Combine(root, "..", "docs", "data", "models.json") }.FirstOrDefault(File.Exists);
		if (path == null)
		{
			return result;
		}

		using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
		foreach (JsonElement m in doc.RootElement.EnumerateArray())
		{
			result[m.GetProperty("id").GetInt64().ToString(CultureInfo.InvariantCulture)] = m.GetProperty("faces").GetInt32();
		}

		return result;
	}

	private static int StlFaceCount(string zip)
	{
		byte[] d = ReadStl(zip);
		string head = Encoding.UTF8.GetString(d, 0, Math.Min(d.Length, 512));
		if (head.TrimStart().StartsWith("solid", StringComparison.Ordinal) && head.Contains("facet", StringComparison.Ordinal))
		{
			return Encoding.UTF8.GetString(d).Split("facet normal").Length - 1;
		}

		return d.Length >= 84 ? (int)Math.Min(BitConverter.ToUInt32(d, 80), int.MaxValue) : 0;
	}

	private static void PrintSummary(string outPath)
	{
		var rows = File.ReadLines(outPath).Skip(1).Select(SplitCsv).ToList();
		int Col(string h) => Array.IndexOf(Header, h);
		Console.WriteLine();
		Console.WriteLine($"=== Summary ({rows.Count} meshes in {outPath}) ===");
		foreach (var g in rows.GroupBy(r => r[Col("class")]).OrderByDescending(g => g.Count()))
		{
			Console.WriteLine($"  {g.Key,-20} {g.Count()}");
		}

		var solids = rows.Where(r => r[Col("class")] == "solid").ToList();
		Console.WriteLine("Checks on solids:");
		foreach (string c in Checks)
		{
			Console.WriteLine($"  {c,-14} " + string.Join("  ", solids.GroupBy(r => r[Col(c)]).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Key} {g.Count()}")));
		}

		var bad = rows.Where(r => r[Col("class")] is "hung" or "crash" or "import_error" || Checks.Any(c => r[Col(c)] is "FAIL" or "timeout")).ToList();
		Console.WriteLine(bad.Count == 0 ? "No failures." : "Failures and timeouts:");
		foreach (var r in bad)
		{
			Console.WriteLine($"  {r[0],-8} {r[1],6}f {r[Col("class")]} {string.Join(" ", Checks.Where(c => r[Col(c)] is not ("ok" or "n/a" or "declined" or "")).Select(c => c + "=" + r[Col(c)]))}: {r[Col("message")]}");
		}
	}

	private static string Fail(List<string> msgs, string message)
	{
		msgs.Add(message);
		return "FAIL";
	}

	private static string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);

	private static string I(int v) => v.ToString(CultureInfo.InvariantCulture);

	public static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;

	private static string[] SplitCsv(string line)
	{
		var cells = new List<string>();
		var cur = new StringBuilder();
		bool quoted = false;
		for (int i = 0; i < line.Length; i++)
		{
			char ch = line[i];
			if (quoted)
			{
				if (ch == '"' && i + 1 < line.Length && line[i + 1] == '"')
				{
					cur.Append('"');
					i++;
				}
				else if (ch == '"')
				{
					quoted = false;
				}
				else
				{
					cur.Append(ch);
				}
			}
			else if (ch == '"')
			{
				quoted = true;
			}
			else if (ch == ',')
			{
				cells.Add(cur.ToString());
				cur.Clear();
			}
			else
			{
				cur.Append(ch);
			}
		}

		cells.Add(cur.ToString());
		return cells.ToArray();
	}
}
