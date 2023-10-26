## MAST Output Structure

```
Processing_Resource
...

Scheduler
...

Scheduling_Server
...

Operation
...

Transaction (
    External_Events
        %external_event
        ...
    Interval_Events
        %internal_event
        ...
    Event_Handlers
        %event_handler
        ...
);
...
```

## MAST elements

### Processing_Resource

Processor 1->1 Processing_Resource

```
Processing_Resource (
   Type                   => Regular_Processor,
   Name                   => %name,
   Max_Interrupt_Priority => 255,
   Min_Interrupt_Priority => 255,
   Worst_ISR_Switch       => 0.00,
   Avg_ISR_Switch         => 0.00,
   Best_ISR_Switch        => 0.00,
   Speed_Factor           => 1.00);
```

### Scheduler

Processor 1->1 Scheduler

```
Scheduler (
   Type            => Primary_Scheduler,
   Name            => %name,
   Host            => %processing_resource,
   Policy          => 
      ( Type                 => Fixed_Priority,
        Worst_Context_Switch => 0.00,
        Avg_Context_Switch   => 0.00,
        Best_Context_Switch  => 0.00,
        Max_Priority         => 254,
        Min_Priority         => 1));
```

```
Scheduler (
   Type            => Primary_Scheduler,
   Name            => %name,
   Host            => %processing_resource,
   Policy          => 
      ( Type                 => EDF,
        Worst_Context_Switch => 0.00,
        Avg_Context_Switch   => 0.00,
        Best_Context_Switch  => 0.00 );
```

### Scheduling_Server

Task 1->1 Scheduling_Server

```
Scheduling_Server (
   Type                       => Regular,
   Name                       => %name,
   Server_Sched_Parameters    => $Server_Sched_Parameter,
   Scheduler                  => %scheduler);
```

**Server_Sched_Parameter**

```
(Type         => Fixed_Priority_Policy,
 The_Priority => %priority,
 Preassigned  => NO)
```

```
(Type         => EDF_policy,
 Deadline     => %deadline,
 Preassigned  => NO)
```

### Operation

Task 1->1 Operation

```
Operation (
   Type                       => Simple,
   Name                       => %name,
   Worst_Case_Execution_Time  => %wcet,
   Avg_Case_Execution_Time    => %acet,
   Best_Case_Execution_Time   => %bcet);
```

### Transaction

Flow 1->1 Transaction

```
Transaction (
   Type            => Regular,
   Name            => %name,
   External_Events => ($External_Event, ...),
   Internal_Events => ($Internal_Event, ...),
   Event_Handlers  => ($Event_Handler, ...);
```

**External_Event**

Flow 1->1 External_Event

```
(Type       => Periodic,
 Name       => %name,
 Period     => %period,
 Max_Jitter => %jitter,
 Phase      => %phase)
```

**Internal_Event**

Task 1->1 Interval_Event

```
(Type => Regular,
 Name => %name)
```

**Event_Handler**

Task 1->1 Event_Handler

```
(Type               => Activity,
 Input_Event        => %input_event,
 Output_Event       => %output_event,
 Activity_Operation => %operation,
 Activity_Server    => %scheduling_server)
```

```
(Type               => Offset,
 Input_Event        => %input_event,
 Output_Event       => %output_event,
 Delay_Max_Interval => %offset_max,
 Delay_Min_Interval => %offset_min,
 Referenced_Event   => %event)
```

```
(Type               => Delay,
 Input_Event        => %input_event,
 Output_Event       => %output_event,
 Delay_Max_Interval => %delay_max,
 Delay_Min_Interval => %delay_min)
```
