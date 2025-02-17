obs = obslua

-- Returns the description displayed in the Scripts window
function script_description()
  return [[Weighted Kuwahara Filter
  Honestly, I don't expect this to work.]]
end
-- Called on script startup
function script_load(settings)
  obs.obs_register_source(source_info)
end

-- Definition of the global variable containing the source_info structure
source_info = {}
source_info.id = 'filter-kuwa-weighted'              -- Unique string identifier of the source type
source_info.type = obs.OBS_SOURCE_TYPE_FILTER   -- INPUT or FILTER or TRANSITION
source_info.output_flags = obs.OBS_SOURCE_VIDEO -- Combination of VIDEO/AUDIO/ASYNC/etc

-- Returns the name displayed in the list of filters
source_info.get_name = function()
  return "New Kuwahara"
end

-- Creates the implementation data for the source
source_info.create = function(settings, source)

  -- Initializes the custom data table
  local data = {}
  data.source = source -- Keeps a reference to this filter as a source object
  data.width = 1       -- Dummy value during initialization phase
  data.height = 1      -- Dummy value during initialization phase

  -- Compiles the effect
  obs.obs_enter_graphics()
  local effect_file_path = script_path() .. 'filter-kuwa-weighted.effect.hlsl'
  data.effect = obs.gs_effect_create_from_file(effect_file_path, nil)
  obs.obs_leave_graphics()

  -- Calls the destroy function if the effect was not compiled properly
  if data.effect == nil then
    obs.blog(obs.LOG_ERROR, "Effect compilation failed for " .. effect_file_path)
    source_info.destroy(data)
    return nil
  end
  -- Retrieves the shader uniform variables
  data.params = {}
  data.params.width = obs.gs_effect_get_param_by_name(data.effect, "width")
  data.params.height = obs.gs_effect_get_param_by_name(data.effect, "height")
  
  data.params.lumcap = obs.gs_effect_get_param_by_name(data.effect, "lumcap")
  data.params.transformangle = obs.gs_effect_get_param_by_name(data.effect, "transformangle")
  data.params.sectors = obs.gs_effect_get_param_by_name(data.effect, "sharpness")
  data.params.box_radius = obs.gs_effect_get_param_by_name(data.effect, "box_radius")
  data.params.intensity = obs.gs_effect_get_param_by_name(data.effect, "intensity")
  data.params.tuning = obs.gs_effect_get_param_by_name(data.effect, "tuning")
  --data.params.pattern_texture = obs.gs_effect_get_param_by_name(data.effect, "pattern_texture")
  --data.params.pattern_size = obs.gs_effect_get_param_by_name(data.effect, "pattern_size")
  --data.params.pattern_gamma = obs.gs_effect_get_param_by_name(data.effect, "pattern_gamma")

  --data.params.palette_texture = obs.gs_effect_get_param_by_name(data.effect, "palette_texture")
  --data.params.palette_size = obs.gs_effect_get_param_by_name(data.effect, "palette_size")
  --data.params.palette_gamma = obs.gs_effect_get_param_by_name(data.effect, "palette_gamma")

  -- Calls update to initialize the rest of the properties-managed settings
  source_info.update(data, settings)
  return data
end

-- Sets the default settings for this source
source_info.get_defaults = function(settings)
  obs.obs_data_set_default_int(settings, "box_radius", 2)
  obs.obs_data_set_default_int(settings, "sharpness", 8)
  
  obs.obs_data_set_default_double(settings, "intensity", .02)
  obs.obs_data_set_default_double(settings, "lumcap", .02)
  
  obs.obs_data_set_default_double(settings, "tuning", 1.0)
  
  obs.obs_data_set_default_double(settings, "transformangle", 0.0)
  --obs.obs_data_set_default_string(settings, "pattern_path", "")
  --obs.obs_data_set_default_double(settings, "pattern_gamma", 1.0)
  --obs.obs_data_set_default_string(settings, "palette_path", "")
  --obs.obs_data_set_default_double(settings, "palette_gamma", 1.0)
end

-- Destroys and release resources linked to the custom data
source_info.destroy = function(data)
  if data.effect ~= nil then
    obs.obs_enter_graphics()
    obs.gs_effect_destroy(data.effect)
    data.effect = nil
    obs.obs_leave_graphics()
  end
end

-- Returns the width of the source
source_info.get_width = function(data)
  return data.width
end

-- Gets the property information of this source
source_info.get_properties = function(data)
  -- Gets the property information of this source
  print("In source_info.get_properties")

  local props = obs.obs_properties_create()
  obs.obs_properties_add_int_slider(props, "sharpness", "Weighting of the standard deviations", 2, 13, 2)
  obs.obs_properties_add_int_slider(props, "box_radius", "Radius of the Kuwahara window", 2, 40, 1)
  obs.obs_properties_add_float_slider(props, "intensity", "Intensity", 0.0, 1.0, .001)
  obs.obs_properties_add_float_slider(props, "lumcap", "Get rid of the black noodling", 0.0, 1.0, .001)
  obs.obs_properties_add_float_slider(props, "transformangle", "Rotation of the kernal", -180.0, 180.0, .5)
  obs.obs_properties_add_float_slider(props, "tuning", "Tuning", 0.0, 3.0, .05)
  return props
end


-- Updates the internal data for this source upon settings change
source_info.update = function(data, settings)
  data.sharpness = obs.obs_data_get_int(settings, "sharpness")
  data.box_radius = obs.obs_data_get_int(settings, "box_radius")
  data.intensity = obs.obs_data_get_double(settings, "intensity")
  data.lumcap = obs.obs_data_get_double(settings, "lumcap")
  data.tuning = obs.obs_data_get_double(settings, "tuning")
  data.transformangle = obs.obs_data_get_double(settings, "transformangle")
   -- Keeps a reference on the settings
  data.settings = settings
end

-- Returns the height of the source
source_info.get_height = function(data)
  return data.height
end

-- Called when rendering the source with the graphics subsystem
source_info.video_render = function(data)
  local parent = obs.obs_filter_get_parent(data.source)
  data.width = obs.obs_source_get_base_width(parent)
  data.height = obs.obs_source_get_base_height(parent)

  obs.obs_source_process_filter_begin(data.source, obs.GS_RGBA, obs.OBS_NO_DIRECT_RENDERING)

  -- Effect parameters initialization goes here

  obs.gs_effect_set_int(data.params.width, data.width)
  obs.gs_effect_set_int(data.params.height, data.height)
  obs.gs_effect_set_int(data.params.box_radius, data.box_radius)
  obs.gs_effect_set_int(data.params.sharpness, data.sharpness)
  obs.gs_effect_set_float(data.params.intensity, data.intensity)
  obs.gs_effect_set_float(data.params.lumcap, data.lumcap)
  obs.gs_effect_set_float(data.params.transformangle, data.transformangle)
  obs.gs_effect_set_float(data.params.tuning, data.tuning)
  obs.obs_source_process_filter_end(data.source, data.effect, data.width, data.height)
end
