// OBS-specific syntax adaptation to HLSL standard to avoid errors reported by the code editor
#define SamplerState sampler_state
#define Texture2D texture2d

// Uniform variables set by OBS (required)
uniform float4x4 ViewProj; // View-projection matrix used in the vertex shader
uniform Texture2D image;   // Texture containing the source picture

// Constants
#define PI 3.141592653589793238
#define PI2 1.57079632679 
#define PI180 0.01745329251 
// General properties
uniform int box_radius = 4;
uniform int sharpness = 8;
// //uniform int box_area = box_radius*box_radius;
uniform float intensity = .6;
uniform float tuning = 1.0;
uniform float lumcap = 0.01;
uniform float transformangle = 0.0;
// Size of the source picture
uniform int width;
uniform int height;

SamplerState linear_wrap
{
    Filter    = Linear; 
    AddressU  = Wrap;
    AddressV  = Wrap;
};
// Interpolation method and wrap mode for sampling a texture
SamplerState linear_clamp
{
    Filter    = Linear;     // Anisotropy / Point / Linear
    AddressU  = Clamp;      // Wrap / Clamp / Mirror / Border / MirrorOnce
    AddressV  = Clamp;      // Wrap / Clamp / Mirror / Border / MirrorOnce
    BorderColor = 00000000; // Used only with Border edges (optional)
};

SamplerState point_clamp
{
    Filter    = Point; 
    AddressU  = Clamp;
    AddressV  = Clamp;
};
// Data type of the input of the vertex shader
struct vertex_data
{
    float4 pos : POSITION;  // Homogeneous space coordinates XYZW
    float2 uv  : TEXCOORD0; // UV coordinates in the source picture
};

// Data type of the output returned by the vertex shader, and used as input 
// for the pixel shader after interpolation for each pixel
struct pixel_data
{
    float4 pos : POSITION;  // Homogeneous screen coordinates XYZW
    float2 uv  : TEXCOORD0; // UV coordinates in the source picture
};

// Vertex shader used to compute position of rendered pixels and pass UV
pixel_data vertex_shader_kuwa(vertex_data vertex)
{
    pixel_data pixel;
    pixel.pos = mul(float4(vertex.pos.xyz, 1.0), ViewProj);
    pixel.uv  = vertex.uv;
    return pixel;
}


//RGB to HCV and back jacked from https://www.chilliant.com/rgb2hsv.html
float Epsilon = .0000000001;
float3 HUEtoRGB(in float H)
  {
    float R = abs(H * 6 - 3) - 1;
    float G = 2 - abs(H * 6 - 2);
    float B = 2 - abs(H * 6 - 4);
    return saturate(float3(R,G,B));
  }
  
float3 HSVtoRGB(in float3 HSV)
  {
    float3 RGB = HUEtoRGB(HSV.x);
    return ((RGB - 1) * HSV.y + 1) * HSV.z;
  }
float3 RGBtoHCV(in float3 RGB)
  {
    // Based on work by Sam Hocevar and Emil Persson
    float4 P = (RGB.g < RGB.b) ? float4(RGB.bg, -1.0, 2.0/3.0) : float4(RGB.gb, 0.0, -1.0/3.0);
    float4 Q = (RGB.r < P.x) ? float4(P.xyw, RGB.r) : float4(RGB.r, P.yzx);
    float C = Q.x - min(Q.w, Q.y);
    float H = abs((Q.w - Q.y) / (6 * C + Epsilon) + Q.z);
    return float3(H, C, Q.x);
  }
float3 RGBtoHSV(in float3 RGB)
  {
    float3 HCV = RGBtoHCV(RGB);
    float S = HCV.y / (HCV.z + Epsilon);
    return float3(HCV.x, S, HCV.z);
  }
 
//Refactored python code of welford's online algorithm
//if it dont work blame https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance 
float4 welford_update(in float4 existing_aggregate)
{

   //count map x
	//mean map y
	//M2 map z
	//new_value map w
    // count += 1
	float delta;
	float delta2;
	float M2;
	float mean;
	float1 count = existing_aggregate.x + 1.0;
	
	//delta = new_value - mean
    delta = (existing_aggregate.w - existing_aggregate.y);
	
    //mean += delta / count
	mean = existing_aggregate.y + (delta/count);
	
    //delta2 = new_value - mean
	delta2 = (existing_aggregate.w - mean);
    
	//M2 += delta * delta2
	M2 = (existing_aggregate.z + (delta * delta2));
    
	//return float4(count, mean, M2, 1.0)
	return float4(count,mean,M2, 1.0);
}

// Retrieve the mean, variance and sample variance from an aggregate
float4 welford_finalize(in float4 existing_aggregate)
{
    //    return (mean, variance, sample_variance)
    return float4(existing_aggregate.y, sqrt(existing_aggregate.z / float(existing_aggregate.x)), sqrt(existing_aggregate.z / float(existing_aggregate.x-1.0)), existing_aggregate.x);
	
}


float1 weighttweak(in float1 stddev)
{
	//return float(float(1.0)/(float(1.0)+float(pow((float(255.0)*stddev*stddev),int(sharpness/int(2))))));
	return float(1.0/(1+stddev));
}

float2 rotation(in float2 position, in float theta)
{
	float sintheta = sin(theta);
	float costheta = cos(theta);
	float newx = costheta*position.x - sintheta*position.y;
	float newy = sintheta*position.x + costheta*position.y;
	return float2(newx,newy);
}

float3 point_tri_dist(in float2 checkpoint, in float intri)
{
	float2 tri1 = float2(0.0,0.86602540378);
	float2 tri2 = float2(-1.0,-0.86602540378);
	float2 tri3 = float2(1.0,-0.86602540378);
	float3 cond = float3(intri,intri,intri);
	cond.x = float((float(distance(checkpoint,tri1)) <= overlap_thresh)*intri);
	cond.y = float((float(distance(checkpoint,tri2)) <= overlap_thresh)*intri);
	cond.z = float((float(distance(checkpoint,tri3)) <= overlap_thresh)*intri);
	return float3(cond.xyz);
}
// float1 area(in float2 point1,in float2 point2,in float2 point3)
// {
	// //A = (1/2) |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
	// return float1(float(0.5)*abs(point1.x*(point2.y-point3.y) + point2.x*(point3.y-point1.y) + point3.x*(point1.y-point2.y)));
// }
float1 signfunc (in float2 point1, in float2 point2, in float2 point3)
{
    return (point1.x - point3.x) * (point2.y - point3.y) - (point2.x - point3.x) * (point1.y - point3.y);
}

// bool PointInTriangle (fPoint pt, fPoint v1, fPoint v2, fPoint v3)
// {
    // float d1, d2, d3;
    // bool has_neg, has_pos;

    // d1 = sign(pt, v1, v2);
    // d2 = sign(pt, v2, v3);
    // d3 = sign(pt, v3, v1);

    // has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    // has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    // return !(has_neg && has_pos);
// }
float3 point_in_unit_tri(in float2 checkpoint)
{
	float deltaangle = float(transformangle*PI180);
	float2 tri1 = rotation(float2( 0.0, 1.0), deltaangle);
	float2 tri2 = rotation(float2( -1.0, -1.0), deltaangle);
	float2 tri3 = rotation(float2( 1.0, -1.0), deltaangle);
	float a1 = signfunc(checkpoint,tri1,tri2);
	float a2 = signfunc(checkpoint,tri2,tri3);
	float a3 = signfunc(checkpoint,tri1,tri3);
	float negative = (((a1 < 0) + (a2 < 0) + (a3 < 0)) > 0.0)? 1.0 : 0.0;
	float positive = (((a1 > 0) + (a2 > 0) + (a3 > 0)) > 0.0)? 1.0 : 0.0;
	float check = (((positive != 1.0) * (negative != 1.0)) < 1.0)? 1.0 : 0.0;
	float intri = check; //1.0; 
	float d1 = float(distance(checkpoint,tri1));
	float d2 = float(distance(checkpoint,tri2));
	float d3 = float(distance(checkpoint,tri3));
	//dist check
	float3 cond = float3(intri,intri,intri);
	
	//cond.x = float((float(d1) <= intensity)*intri);
	//cond.y = float((float(d2) <= intensity)*intri);
	//cond.z = float((float(d3) <= intensity)*intri);
	
	cond.x = float((d1 <= d2) * (d1 <= d3) * intri);
	cond.y = float((d2 <= d1) * (d2 <= d3) * intri);
	cond.z = float((d3 <= d2) * (d3 <= d1) * intri);
	return float3(cond.xyz);
}


float4 eigen_vector_to_angle(in float2 eigenvector)
{
	float2 normaleigen = normalize(eigenvector.xy); 
	float1 angle = atan2(normaleigen.x,normaleigen.y);
	return float4(angle,normaleigen.xy,1.0);

}

float2 rotate_point(in float2 position, in float1 angle)
{
	float1 x = cos(angle)*position.x - sin(angle)*position.y;
	float1 y = sin(angle)*position.x + cos(angle)*position.y;
	return float2(x,y);
}

float2 pull_scales(in int2 intoff)
{	
	float vert = ((intoff.y % 2 + 1) * -1 * (intoff.x-1));
	float horiz = ((intoff.x % 2 + 1) * -1 * (intoff.y-1));
	return float2(horiz,vert);
}
// approximate derivative of the image at the point for Anisotropy
// blame https://en.wikipedia.org/wiki/Sobel_operator
float4 sobel_convolution(in float2 position)
{
	float2 fulloff;
	float3 horizcomp = float3(0.0,0.0,0.0);
	float3 vertcomp = float3(0.0,0.0,0.0);
	float2 currscale = float2(1.0,1.0);
	float3 horizslope;
	float3 vertslope;
	float4 temp;
	int2 intoff;
	int i = 0;
	float arclength;
	[unroll(9)] while(i < 9)
	{
		
		intoff.x = floor(i / float(3.0));
		intoff.y = i % 3;
		
		currscale = float2(pull_scales(intoff));
		
		fulloff.x = position.x + ((float(intoff.x) - 1.0)/ width);
		fulloff.y = position.y + ((float(intoff.y) - 1.0)/ height);
		
		temp = image.Sample(linear_clamp, fulloff);
		
		horizcomp.x = horizcomp.x + (temp.x * currscale.x);
		horizcomp.y = horizcomp.y + (temp.y * currscale.x);
		horizcomp.z = horizcomp.z + (temp.z * currscale.x);
		
		vertcomp.x = vertcomp.x + (temp.x * currscale.y);
		vertcomp.y = vertcomp.y + (temp.y * currscale.y);
		vertcomp.z = vertcomp.z + (temp.z * currscale.y);
		i = i+1;
	}
	horizslope = float3(float(0.25)*horizcomp.x,float(0.25)*horizcomp.y,float(0.25)*horizcomp.z);
	vertslope = float3(float(0.25)*vertcomp.x,float(0.25)*vertcomp.y,float(0.25)*vertcomp.z);
	//structure tensor
	float E = dot(horizslope,horizslope);
	float F = dot(vertslope,horizslope);
	float G = dot(vertslope,vertslope);
	//(horizslope.x+horizslope.y+horizslope.z)/3.0,(vertslope.x+vertslope.y+vertslope.z)/3.0
	return float4(E,F,G,1.0);
}

float4 eigenvalue(in float3 structure_tensor)
{
	float E = structure_tensor.x;
	float F = structure_tensor.y;
	float G = structure_tensor.z;
	float eigen1;
	float eigen2;
	float anisotropy;
	float alpha;
	float beta;
	float eplusg = E + G;
	float eminusgsquared = E*E - 2.0*G*E + G*G;
	float fourfsquared = float(4.0)*F*F;
	eigen1 = float(0.5)*(eplusg + sqrt(eminusgsquared + fourfsquared));
	eigen2 = float(0.5)*(eplusg - sqrt(eminusgsquared + fourfsquared));
	anisotropy = ((eigen1+eigen2) > 0.0)? abs(float(eigen1-eigen2)/float(eigen1+eigen2)) : 0.0;
	alpha = ((tuning + anisotropy)/tuning)*float(box_radius);
	beta = (tuning/(tuning + anisotropy))*float(box_radius);

	return float4(alpha,beta,eigen1 - E,anisotropy);
}




// float1 base_weight(in float2 position)
// {
// float etayy = eta*position.y*position.y;
// float xpluszeta = position.x+zeta;
// float step1 = xpluszeta-etayy;
// float step2 = step1*step1;
// float condition = (x >= etayy-zeta);
// return float(step2*condition);
// }
float4 sortthree(in float4 val1, in float4 val2, in float4 val3)
{
	float4 compare1 = val1;
	if (compare1.w > val2.w)
	{
	compare1 = val2;
	}
	if (compare1.w > val3.w)
	{
	compare1 = val3;
	}
	return float4(compare1.xyzw);
}
float4 find_read_window(in float1 windowwidth, in float1 windowheight, in float1 angle)
{
	float width = abs(sin(PI2 + angle)*windowwidth) + abs(sin(-1.0*angle)*windowheight);
	float height = abs(sin(PI2 + angle)*windowheight) + abs(sin(-1.0*angle)*windowwidth);
	width = (width>=2.0)? width : 2.0;
	height =(height>=2.0)? height : 2.0;
	return float4(width,height,1.0,1.0);


}
float4 core_logic(in float2 position)
{
	float4 sobelout = sobel_convolution(position).xyzw;
	float E = sobelout.x;
	float F = sobelout.y;
	float G = sobelout.z;
	float hout = sobelout.w;
	float4 eigenout = eigenvalue(float3(E,F,G));
	float alpha = eigenout.x;
	float beta = eigenout.y;
	float anisotropy = eigenout.w;
	float2 eigenvector = float2(eigenout.z,F*float(-1.0));
	float angle = eigen_vector_to_angle(eigenvector).x;
	//begin sampling
	float x;
	float y;
	float currlum;
	float4 zone1avg = float4(0.0,0.0,0.0,0.0);
	float4 zone2avg = float4(0.0,0.0,0.0,0.0);
	float4 zone3avg = float4(0.0,0.0,0.0,0.0);
	float currentstdlum = 0.0;
	float4 temp;
	float4 temp2;
	float4 aggregatetemp = float4 (0.0,0.0,0.0,0.0);
	float2 fulloff;
	float2 mappedoff;
	float3 distances;
	float pointintri;
	float4 zone1 = float4 (0.0,0.0,0.0,0.0);
	float4 zone2 = float4 (0.0,0.0,0.0,0.0);
	float4 zone3 = float4 (0.0,0.0,0.0,0.0);
	float zone1weight;
	float zone2weight;
	float zone3weight;
	float4 totalavg;
	int i = 0;
	float4 blur = float4(0.0,0.0,0.0,0.0);
	float windowwidth = alpha;
	float windowheight = beta;
	float2 truewindow = find_read_window(windowwidth,windowheight,angle).xy;
	//float2 truewindow = float2(box_radius,box_radius);
	float2 scale = float2(float(1.0)/(float(float(2.0)*windowwidth)),float(1.0)/(float(float(2.0)*windowheight)));
	
	[unroll(800)] while (i < truewindow.x*truewindow.y)
	{
		x = float(floor(float(i) / float(truewindow.y))) - float(0.5)*float(truewindow.x);
		y = (i % truewindow.y) - float(0.5)*float(truewindow.y);
		i++;
		fulloff.x = position.x - (x / float(width));
		fulloff.y = position.y - (y / float(height));
		temp = image.Sample(linear_clamp, fulloff);
		currlum = RGBtoHSV(temp.xyz).z;
		//mappedoff = rotation(float2(x,y),angle);
		mappedoff = float2(x,y);
		mappedoff = float2(mappedoff.x/truewindow.x,mappedoff.y/truewindow.y);
		distances = point_in_unit_tri(mappedoff);
		
		
		blur = float4(blur.x+temp.x*any(distances == 1.0),blur.y+temp.y*any(distances == 1.0),blur.z+temp.z*any(distances == 1.0),blur.w+1.0*any(distances == 1.0));
		
		zone1 = float4(zone1.x+temp.x*distances.x,zone1.y+temp.y*distances.x,zone1.z+temp.z*distances.x,1.0);
		temp2 = welford_update(float4(zone1avg.xyz,currlum));
		zone1avg = float4(zone1avg.x*(float(1.0)-distances.x)+temp2.x*distances.x,zone1avg.y*(float(1.0)-distances.x)+temp2.y*distances.x,zone1avg.z*(float(1.0)-distances.x)+temp2.z*distances.x,1.0);
		
		temp2 = welford_update(float4(zone2avg.xyz,currlum));
		zone2avg = float4(zone2avg.x*(float(1.0)-distances.y)+temp2.x*distances.y,zone2avg.y*(float(1.0)-distances.y)+temp2.y*distances.y,zone2avg.z*(float(1.0)-distances.y)+temp2.z*distances.y,1.0);
		zone2 = float4(zone2.x+temp.x*distances.y,zone2.y+temp.y*distances.y,zone2.z+temp.z*distances.y,1.0);
		
		temp2 = welford_update(float4(zone3avg.xyz,currlum));
		zone3avg = float4(zone3avg.x*(float(1.0)-distances.z)+temp2.x*distances.z,zone3avg.y*(float(1.0)-distances.z)+temp2.y*distances.z,zone3avg.z*(float(1.0)-distances.z)+temp2.z*distances.z,1.0);
		zone3 = float4(zone3.x+temp.x*distances.z,zone3.y+temp.y*distances.z,zone3.z+temp.z*distances.z,1.0);
		
	}
	//end loop, finalize values and weight them
	zone1avg = welford_finalize(zone1avg);
	zone1 = zone1 / float(zone1avg.w);
	zone1weight = weighttweak(zone1avg.y);
	zone2avg = welford_finalize(zone2avg);
	zone2 = zone2 / zone2avg.w;
	zone2weight = weighttweak(zone2avg.y);
	zone3avg = welford_finalize(zone3avg);
	zone3 = zone3 / zone3avg.w;
	zone3weight = weighttweak(zone3avg.y);
	
	blur = float4(blur.xyz/blur.w,1.0);
	float totalweight = zone1weight+zone2weight+zone3weight;
	float avgr = float((zone1.x*zone1weight+zone2.x*zone2weight+zone3.x*zone3weight)/float(totalweight));
	float avgg = float((zone1.y*zone1weight+zone2.y*zone2weight+zone3.y*zone3weight)/float(totalweight));
	float avgb = float((zone1.z*zone1weight+zone2.z*zone2weight+zone3.z*zone3weight)/float(totalweight));
	totalavg = float4(avgr,avgg,avgb,1.0);
	//totalavg = sortthree(float4(zone1.xyz,zone1weight),float4(zone2.xyz,zone2weight),float4(zone3.xyz,zone3weight));
	//totalavg = (zone1 + zone2 + zone3)/float(zone1avg.w+zone2avg.w+zone3avg.w);
	//float checker = (distance(float2(0.0,0.0),float2(G,hout)) != 0)? 1.0: 0.0;
	if (all(totalavg.xyz <= intensity))
	{
		totalavg = image.Sample(linear_clamp, position);
	}
	return float4(totalavg.xyz,1.0);
	//return float4(clamp(truewindow.x,0.0,2*box_radius),clamp(truewindow.y,0.0,2*box_radius),0.0,1.0);
}

float4 pixel_shader_kuwa(pixel_data pixel) : TARGET
{
    //float4 chosenkuwa = get_neighbor_regions(pixel.uv);
	float4 chosenkuwa = core_logic(pixel.uv);
	//float4 chosenkuwa = float4((RGBtoHSV(float3(image.Sample(linear_clamp, pixel.uv).xyz))).zzz,1.0);
	//float4 chosenkuwa = float4(1.0,0.5,0.0,1.0);
	if ((RGBtoHSV(chosenkuwa.xyz).z == 0.0))
	{
		chosenkuwa = image.Sample(linear_clamp, pixel.uv);
	}
    return float4(chosenkuwa.xyz,1.0);
}

technique Draw
{
    pass
    {
        vertex_shader = vertex_shader_kuwa(vertex);
        pixel_shader  = pixel_shader_kuwa(pixel);
    }
}
